"""
recorder.py
-----------
Option C: Record every hand the user plays (or the bot plays) and save
the (state, action, outcome) tuples to a local replay buffer.

The recorder runs as a background thread alongside the engine:
  1. Each actionable frame → record (state, action)
  2. Each result frame → record outcome (win/loss/push)
  3. Saves JSONL to data/replay_buffer.jsonl (append mode)

The replay buffer is used to:
  a. Fine-tune the Q-table offline (bj-assistant train-ai --from-replay)
  b. Compute real session statistics (EV, decision accuracy vs BS)
  c. Detect patterns (which hands lose most, when deviations help)

Usage (automatic — engine.py starts it if enabled):
    recorder = HandRecorder(enabled=True)
    recorder.record_action(gf, decision)   # called on each playing frame
    recorder.record_result(outcome)        # called on each result frame
    recorder.flush()                       # saves pending buffer on stop
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

log = logging.getLogger(__name__)

# Default replay buffer path
DEFAULT_BUFFER_PATH = Path(__file__).parent.parent / "data" / "replay_buffer.jsonl"


class HandRecorder:
    """
    Records (state, action, outcome) tuples for every BJ hand.
    Thread-safe — can be called from the engine loop.
    """

    def __init__(
        self,
        enabled: bool = True,
        buffer_path: Optional[Path] = None,
        flush_every: int = 10,    # flush to disk every N hands
    ) -> None:
        self._enabled    = enabled
        self._path       = Path(buffer_path) if buffer_path else DEFAULT_BUFFER_PATH
        self._flush_every = flush_every
        self._lock       = threading.Lock()
        self._pending: List[dict] = []
        self._hands_recorded = 0
        self._current_hand: Optional[dict] = None
        self._session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

        if enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            log.info("HandRecorder: recording to %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_action(
        self,
        player_total:  int,
        is_soft:       bool,
        dealer_upcard: str,
        true_count:    float,
        action_taken:  str,      # "H"/"S"/"D"/"P"
        bs_action:     str,      # Basic Strategy recommendation
        ai_action:     Optional[str],
        can_double:    bool = True,
        can_split:     bool = False,
        balance:       Optional[int] = None,
        bet_amount:    int = 250,
    ) -> None:
        """Record the action taken on an actionable frame."""
        if not self._enabled:
            return

        record = {
            "session": self._session_id,
            "ts":      time.time(),
            "type":    "action",
            "player_total":  player_total,
            "is_soft":       is_soft,
            "dealer_upcard": dealer_upcard,
            "true_count":    round(true_count, 2),
            "action_taken":  action_taken,
            "bs_action":     bs_action,
            "ai_action":     ai_action,
            "followed_bs":   action_taken == bs_action,
            "can_double":    can_double,
            "can_split":     can_split,
            "balance":       balance,
            "bet_amount":    bet_amount,
        }
        with self._lock:
            self._current_hand = record

    def record_result(
        self,
        outcome: str,            # "win"/"loss"/"push"/None
        payout_multiplier: float = 1.0,
    ) -> None:
        """Record the hand outcome when result screen is detected."""
        if not self._enabled or self._current_hand is None:
            return

        with self._lock:
            hand = dict(self._current_hand)
            hand["outcome"]   = outcome
            hand["payout_multiplier"] = payout_multiplier
            # Reward for RL: +1.5 BJ, +1 win, 0 push, -1 loss
            reward_map = {"win": 1.0, "loss": -1.0, "push": 0.0}
            hand["reward"]    = reward_map.get(outcome or "", 0.0) * payout_multiplier
            self._pending.append(hand)
            self._current_hand = None
            self._hands_recorded += 1

        if self._hands_recorded % self._flush_every == 0:
            self._flush_to_disk()

    def record_raw_hand(self, hand_dict: dict) -> None:
        """Record a complete hand dict directly (for testing/replay)."""
        if not self._enabled:
            return
        with self._lock:
            self._pending.append(hand_dict)

    def flush(self) -> int:
        """Flush all pending records to disk. Returns number of records written."""
        return self._flush_to_disk()

    @property
    def hands_recorded(self) -> int:
        return self._hands_recorded

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def session_stats(self) -> dict:
        """Compute stats from the current session's pending buffer."""
        with self._lock:
            buf = [r for r in self._pending if r.get("outcome")]

        if not buf:
            return {}

        total   = len(buf)
        wins    = sum(1 for r in buf if r["outcome"] == "win")
        losses  = sum(1 for r in buf if r["outcome"] == "loss")
        pushes  = sum(1 for r in buf if r["outcome"] == "push")
        ev      = sum(r["reward"] for r in buf) / total if total else 0
        bs_acc  = sum(1 for r in buf if r.get("followed_bs")) / total if total else 0

        return {
            "hands":    total,
            "wins":     wins,
            "losses":   losses,
            "pushes":   pushes,
            "win_rate": wins / total if total else 0,
            "ev":       round(ev, 4),
            "bs_accuracy": round(bs_acc, 4),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_to_disk(self) -> int:
        with self._lock:
            to_write = list(self._pending)
            self._pending.clear()

        if not to_write:
            return 0

        try:
            with open(self._path, "a") as f:
                for record in to_write:
                    f.write(json.dumps(record) + "\n")
            log.debug("HandRecorder: flushed %d records to %s", len(to_write), self._path)
        except Exception as exc:
            log.error("HandRecorder: flush failed: %s", exc)
            # Put records back
            with self._lock:
                self._pending = to_write + self._pending
            return 0

        return len(to_write)


# ---------------------------------------------------------------------------
# Replay buffer reader (for offline training)
# ---------------------------------------------------------------------------

def load_replay_buffer(path: Optional[Path] = None) -> List[dict]:
    """Load all records from the replay buffer JSONL file."""
    p = Path(path) if path else DEFAULT_BUFFER_PATH
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def replay_stats(path: Optional[Path] = None) -> dict:
    """Compute aggregate stats from the full replay buffer."""
    records = load_replay_buffer(path)
    completed = [r for r in records if r.get("outcome")]

    if not completed:
        return {"total_records": len(records), "completed_hands": 0}

    total   = len(completed)
    wins    = sum(1 for r in completed if r["outcome"] == "win")
    losses  = sum(1 for r in completed if r["outcome"] == "loss")
    pushes  = sum(1 for r in completed if r["outcome"] == "push")
    ev      = sum(r.get("reward", 0) for r in completed) / total
    bs_acc  = sum(1 for r in completed if r.get("followed_bs")) / total

    # Per true-count-bucket EV
    tc_ev: dict = {}
    for r in completed:
        tc = r.get("true_count", 0)
        bucket = int(round(tc))
        if bucket not in tc_ev:
            tc_ev[bucket] = []
        tc_ev[bucket].append(r.get("reward", 0))
    tc_ev_avg = {k: round(sum(v)/len(v), 4) for k, v in sorted(tc_ev.items())}

    return {
        "total_records":  len(records),
        "completed_hands": total,
        "wins":   wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate":    round(wins/total, 4),
        "ev_per_hand": round(ev, 4),
        "bs_accuracy": round(bs_acc, 4),
        "ev_by_tc":    tc_ev_avg,
        "sessions":    len(set(r.get("session","") for r in records)),
    }
