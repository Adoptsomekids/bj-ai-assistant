"""
ai_advisor.py
-------------
Runtime AI decision layer.

Decision cascade (fastest → best available):
  1. Q-table lookup (Monte Carlo trained) — if model loaded and state covered
  2. Basic Strategy + Illustrious 18 + Hi-Lo deviations (always available)

The Q-table is trained offline via mc_trainer.py. If no model file exists,
the advisor silently falls back to Basic Strategy (no crash).

Integration in engine.py:
    advisor = AIAdvisor.load("models/bj_qtable.npy")
    ai_action = advisor.decide(player_total, is_soft, dealer_upcard, tc, can_double, can_split)
    # ai_action is a strategy code: "H", "S", "D", "P"
    # It overrides or confirms the Basic Strategy decision.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .mc_trainer import (
    ACTIONS, ACTION_STAND, ACTION_HIT, ACTION_DOUBLE, ACTION_SPLIT,
    TOTAL_STATES, NUM_ACTIONS, encode_state,
    PLAYER_TOTAL_MIN, PLAYER_TOTAL_MAX, RANK_VALUES
)

log = logging.getLogger(__name__)

# Default model path (relative to bj-ai-assistant/)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "bj_qtable.npy"


class AIAdvisor:
    """
    Runtime BJ advisor backed by a trained Q-table.

    Falls back gracefully to None (caller uses Basic Strategy) if:
    - Model file doesn't exist
    - State is out of trained range
    - NumPy not available
    """

    def __init__(self, q_table: Optional[np.ndarray] = None) -> None:
        self._q_table = q_table
        self._loaded  = q_table is not None
        if self._loaded:
            log.info("AIAdvisor: Q-table loaded (%d states × %d actions)",
                     q_table.shape[0], q_table.shape[1])
        else:
            log.info("AIAdvisor: no Q-table — using Basic Strategy only")

    @classmethod
    def load(cls, model_path: Optional[str | Path] = None) -> "AIAdvisor":
        """
        Load Q-table from disk. Returns a fallback-only advisor if file not found.
        """
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            log.warning("AIAdvisor: model not found at %s — Basic Strategy fallback only. "
                        "Run: python -m bj_assistant.ai.mc_trainer to train.", path)
            return cls(q_table=None)
        try:
            q = np.load(str(path))
            log.info("AIAdvisor: loaded Q-table from %s (%.1f KB)", path, path.stat().st_size / 1024)
            return cls(q_table=q)
        except Exception as exc:
            log.error("AIAdvisor: failed to load Q-table: %s", exc)
            return cls(q_table=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_model(self) -> bool:
        return self._loaded

    def decide(
        self,
        player_total:  int,
        is_soft:       bool,
        dealer_upcard: str,
        true_count:    float,
        can_double:    bool = True,
        can_split:     bool = False,
    ) -> Optional[str]:
        """
        Return an action code ("H"/"S"/"D"/"P") from the Q-table,
        or None if the Q-table is unavailable / state out of range.

        Parameters
        ----------
        player_total : int
            Player's hand total (4–21).
        is_soft : bool
            True if the hand is soft (contains an Ace counted as 11).
        dealer_upcard : str
            Dealer's visible card rank: "2"–"9", "10", "J", "Q", "K", "A".
        true_count : float
            Current Hi-Lo true count.
        can_double : bool
        can_split : bool

        Returns
        -------
        str or None
            Strategy action code, or None to fall back to Basic Strategy.
        """
        if not self._loaded:
            return None

        # Validate state
        if not (PLAYER_TOTAL_MIN <= player_total <= PLAYER_TOTAL_MAX):
            return None

        # Convert dealer upcard to integer (A=11, face cards=10)
        d_val = RANK_VALUES.get(dealer_upcard, 10)
        if dealer_upcard == 'A':
            d_val = 11

        try:
            state_idx = encode_state(
                player_total, is_soft, d_val, true_count, can_double, can_split
            )
            q_vals = self._q_table[state_idx]

            # Mask out unavailable actions
            masked = q_vals.copy()
            if not can_double:
                masked[ACTION_DOUBLE] = -999.0
            if not can_split:
                masked[ACTION_SPLIT]  = -999.0

            best_action_idx = int(np.argmax(masked))
            action_code = ACTIONS[best_action_idx]

            log.debug(
                "AIAdvisor: total=%d%s dealer=%s TC=%.1f → %s  (Q=%+.3f)",
                player_total, "s" if is_soft else "",
                dealer_upcard, true_count,
                action_code, masked[best_action_idx]
            )
            return action_code

        except (IndexError, Exception) as exc:
            log.debug("AIAdvisor: state lookup failed: %s", exc)
            return None

    def q_values_for_display(
        self,
        player_total:  int,
        is_soft:       bool,
        dealer_upcard: str,
        true_count:    float,
        can_double:    bool = True,
        can_split:     bool = False,
    ) -> Optional[dict]:
        """
        Return Q-values dict for all actions — used by HUD for confidence display.

        Returns
        -------
        dict like {"H": 0.32, "S": -0.15, "D": 0.41, "P": -0.20}
        or None if unavailable.
        """
        if not self._loaded:
            return None
        try:
            d_val = RANK_VALUES.get(dealer_upcard, 10)
            if dealer_upcard == 'A':
                d_val = 11
            state_idx = encode_state(
                player_total, is_soft, d_val, true_count, can_double, can_split
            )
            q_vals = self._q_table[state_idx]
            return {ACTIONS[i]: float(q_vals[i]) for i in range(NUM_ACTIONS)}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Singleton-style global instance (lazy init)
    # ------------------------------------------------------------------

    _instance: Optional["AIAdvisor"] = None

    @classmethod
    def get_instance(cls) -> "AIAdvisor":
        """Return the global AIAdvisor instance, loading from default path."""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance
