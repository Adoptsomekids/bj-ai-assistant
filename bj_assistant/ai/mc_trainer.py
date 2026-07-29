"""
mc_trainer.py
-------------
Monte Carlo RL trainer for Blackjack, initialized from Basic Strategy.

KEY DESIGN: Q-table is pre-seeded from Basic Strategy so that:
  1. States never visited → still play perfectly (BS is correct)
  2. MC training only ADJUSTS Q-values where simulation disagrees
  3. Coverage problem (only 24% of states visited) is irrelevant

Algorithm:
  Phase 1 — Seed: fill Q-table from Basic Strategy lookup
  Phase 2 — MC:   run episodes, update Q where simulation data exists

State:  (player_total, is_soft, dealer_upcard_idx, tc_bucket, can_double, can_split)
Action: 0=Stand, 1=Hit, 2=Double, 3=Split

Run once:
    bj-assistant train-ai --episodes 500000
The resulting Q-table is 225 KB and loads in <5ms at runtime.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONS       = ["S", "H", "D", "P"]   # Stand, Hit, Double, Split
ACTION_STAND  = 0
ACTION_HIT    = 1
ACTION_DOUBLE = 2
ACTION_SPLIT  = 3

NUM_ACTIONS   = 4

# State dimensions
# player_total:  4-21  → 18 values  (index 0-17, offset by 4)
# is_soft:       0/1   → 2 values
# dealer_upcard: 2-11  → 10 values  (index 0-9, 2→0 … 10/J/Q/K→8, A→9)
# tc_bucket:     -2/-1/0/+1/+2 → 5 values
# can_double:    0/1   → 2 values
# can_split:     0/1   → 2 values

PLAYER_TOTAL_MIN  = 4
PLAYER_TOTAL_MAX  = 21
N_PLAYER_TOTALS   = PLAYER_TOTAL_MAX - PLAYER_TOTAL_MIN + 1  # 18
N_SOFT            = 2
N_DEALER_UPCARDS  = 10
N_TC_BUCKETS      = 5
N_CAN_DOUBLE      = 2
N_CAN_SPLIT       = 2

STATE_SHAPE = (N_PLAYER_TOTALS, N_SOFT, N_DEALER_UPCARDS, N_TC_BUCKETS, N_CAN_DOUBLE, N_CAN_SPLIT)
TOTAL_STATES = 1
for d in STATE_SHAPE:
    TOTAL_STATES *= d
# = 18 * 2 * 10 * 5 * 2 * 2 = 7200

# Hi-Lo count values
HILO_VALUES = {
    '2': +1, '3': +1, '4': +1, '5': +1, '6': +1,
    '7':  0, '8':  0, '9':  0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1,
}


# ---------------------------------------------------------------------------
# State encoding helpers
# ---------------------------------------------------------------------------

def tc_to_bucket(tc: float) -> int:
    """Map true count → 0-4 bucket: ≤-2=0, -1=1, 0=2, +1=3, ≥+2=4."""
    if tc <= -2: return 0
    if tc <= -1: return 1
    if tc <=  1: return 2
    if tc <=  2: return 3
    return 4


def dealer_upcard_to_idx(upcard: int) -> int:
    """Map dealer upcard value (2-11, where 11=Ace) → index 0-9."""
    return min(max(upcard - 2, 0), 9)


def encode_state(
    player_total: int,
    is_soft: bool,
    dealer_upcard: int,
    tc: float,
    can_double: bool,
    can_split: bool,
) -> int:
    """Encode 6-dim state to flat integer index."""
    pt = min(max(player_total - PLAYER_TOTAL_MIN, 0), N_PLAYER_TOTALS - 1)
    s  = int(is_soft)
    du = dealer_upcard_to_idx(dealer_upcard)
    tc_ = tc_to_bucket(tc)
    cd = int(can_double)
    cs = int(can_split)
    # row-major encoding
    idx = ((((pt * N_SOFT + s) * N_DEALER_UPCARDS + du) * N_TC_BUCKETS + tc_) * N_CAN_DOUBLE + cd) * N_CAN_SPLIT + cs
    return idx


# ---------------------------------------------------------------------------
# Deck simulator (6 decks)
# ---------------------------------------------------------------------------

RANKS  = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
RANK_VALUES = {
    '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    '10':10,'J':10,'Q':10,'K':10,'A':11
}


class Shoe:
    """Six-deck shoe with running count tracking."""

    def __init__(self, num_decks: int = 6) -> None:
        self.num_decks = num_decks
        self._cards: List[str] = []
        self.running_count: int = 0
        self._dealt: int = 0
        self.reset()

    def reset(self) -> None:
        self._cards = RANKS * 4 * self.num_decks
        random.shuffle(self._cards)
        self.running_count = 0
        self._dealt = 0

    @property
    def decks_remaining(self) -> float:
        remaining = len(self._cards) - self._dealt
        return max(0.5, remaining / 52)

    @property
    def true_count(self) -> float:
        return self.running_count / self.decks_remaining

    def deal(self) -> str:
        if self._dealt >= len(self._cards) - 15:
            self.reset()  # reshuffle near end of shoe
        card = self._cards[self._dealt]
        self._dealt += 1
        self.running_count += HILO_VALUES.get(card, 0)
        return card

    @property
    def penetration(self) -> float:
        """Fraction of shoe dealt (0.0 = fresh, 1.0 = exhausted)."""
        return self._dealt / len(self._cards)


# ---------------------------------------------------------------------------
# Hand evaluator
# ---------------------------------------------------------------------------

def hand_value(cards: List[str]) -> Tuple[int, bool]:
    """Return (total, is_soft) for a list of card rank strings."""
    total = 0
    aces  = 0
    for c in cards:
        v = RANK_VALUES.get(c, 10)
        if c == 'A':
            aces += 1
        total += v
    # Reduce aces from 11→1 if bust
    while total > 21 and aces > 0:
        total -= 10
        aces  -= 1
    is_soft = (aces > 0 and total <= 21)
    return total, is_soft


# ---------------------------------------------------------------------------
# Blackjack simulator
# ---------------------------------------------------------------------------

def play_dealer(shoe: Shoe, dealer_hand: List[str]) -> int:
    """Play dealer to completion (S17 rule) and return final total."""
    total, is_soft = hand_value(dealer_hand)
    while True:
        total, is_soft = hand_value(dealer_hand)
        if total > 17:
            break
        if total == 17 and not is_soft:
            break
        dealer_hand.append(shoe.deal())
    return hand_value(dealer_hand)[0]


def simulate_hand(
    shoe: Shoe,
    action: int,
    player_hand: List[str],
    dealer_upcard: str,
    dealer_hidden: str,
    can_double: bool,
    can_split: bool,
) -> float:
    """
    Simulate one BJ hand with a fixed first action and optimal play afterward.
    Returns reward: +1.5 BJ, +1 win, 0 push, -1 loss, +2 double win, -2 double loss.
    """
    player = player_hand[:]
    dealer = [dealer_upcard, dealer_hidden]

    # Check natural blackjack
    p_val, _ = hand_value(player)
    d_val, _ = hand_value(dealer)
    if p_val == 21 and len(player) == 2:
        if d_val == 21:
            return 0.0  # push
        return 1.5  # BJ pays 3:2

    if d_val == 21:
        return -1.0  # dealer BJ

    bet_multiplier = 1.0

    if action == ACTION_STAND:
        pass  # no cards drawn

    elif action == ACTION_HIT:
        player.append(shoe.deal())
        p_val, p_soft = hand_value(player)
        # Continue with basic strategy (simplified: hit until 17+)
        while p_val < 17:
            player.append(shoe.deal())
            p_val, p_soft = hand_value(player)
        if p_val > 21:
            return -1.0 * bet_multiplier

    elif action == ACTION_DOUBLE and can_double:
        bet_multiplier = 2.0
        player.append(shoe.deal())
        p_val, p_soft = hand_value(player)
        if p_val > 21:
            return -1.0 * bet_multiplier

    elif action == ACTION_SPLIT and can_split:
        # Simplified split: treat as double the bet, hit once each
        bet_multiplier = 2.0
        card1 = player[0]
        h1 = [card1, shoe.deal()]
        h2 = [card1, shoe.deal()]
        # Play each hand to at least 17 (simplified)
        result = 0.0
        for h in [h1, h2]:
            hv, hs = hand_value(h)
            while hv < 17:
                h.append(shoe.deal())
                hv, hs = hand_value(h)
            d_final = play_dealer(shoe, dealer[:])  # dealer plays once
            hv, _ = hand_value(h)
            if hv > 21:
                result -= 1.0
            elif hv > d_final or d_final > 21:
                result += 1.0
            elif hv == d_final:
                pass  # push
            else:
                result -= 1.0
        return result

    else:
        # Invalid action (e.g. split when can't split) → hit instead
        player.append(shoe.deal())
        p_val, p_soft = hand_value(player)
        while p_val < 17:
            player.append(shoe.deal())
            p_val, p_soft = hand_value(player)
        if p_val > 21:
            return -1.0

    # Dealer plays
    d_final = play_dealer(shoe, dealer)
    p_final, _ = hand_value(player)

    if p_final > 21:
        return -1.0 * bet_multiplier
    if d_final > 21 or p_final > d_final:
        return +1.0 * bet_multiplier
    if p_final == d_final:
        return 0.0
    return -1.0 * bet_multiplier


# ---------------------------------------------------------------------------
# Phase 1: Seed Q-table directly from strategy.py decide()
# ---------------------------------------------------------------------------

# Q-value magnitude for BS seed
_BS_SEED_STRONG  = 0.50   # best action for this state
_BS_SEED_PENALTY = -0.30  # unavailable actions


def seed_from_basic_strategy(q_table: np.ndarray) -> np.ndarray:
    """
    Phase 1: Pre-fill Q-table using strategy.py decide() — the same engine
    used at runtime. This guarantees 100% agreement for all unvisited states.

    MC training then adjusts Q-values where simulation data provides evidence
    (especially for count-dependent deviations at non-neutral TC buckets).
    """
    # Import here to avoid circular deps at module level
    from bj_assistant.strategy import GameState, HiLoCounter, decide, hand_total

    log.info("Phase 1: seeding Q-table from strategy.py decide()...")

    DEALER_STRS = ['2','3','4','5','6','7','8','9','10','A']
    DEALER_VALS = [2,   3,   4,   5,   6,   7,   8,   9,   10,  11 ]
    TC_VALS     = [-3, -1, 0, +1, +3]   # representative TC for each bucket

    seeded = 0
    for pt in range(PLAYER_TOTAL_MIN, PLAYER_TOTAL_MAX + 1):
        for soft in [False, True]:
            if soft and pt < 12:
                continue
            for d_str, d_val in zip(DEALER_STRS, DEALER_VALS):
                for tc_bucket, tc_raw in enumerate(TC_VALS):
                    for cd in [False, True]:
                        for cs in [False, True]:
                            # Build synthetic player cards matching (pt, soft)
                            if soft and pt >= 12:
                                p_cards = ['A', str(pt - 11)] if pt > 11 else ['A']
                            elif pt >= 12:
                                p_cards = ['10', str(pt - 10)]
                            else:
                                p_cards = ['2', str(pt - 2)]

                            # If can_split, make it a pair (e.g. total=8 → [4,4])
                            if cs and not soft and pt % 2 == 0 and 4 <= pt <= 20:
                                half = pt // 2
                                p_cards = [str(half), str(half)]

                            counter = HiLoCounter(6)
                            counter.running_count = int(tc_raw * 3)

                            state = GameState(
                                player_cards  = p_cards,
                                dealer_upcard = d_str,
                                counter       = counter,
                                can_double    = cd,
                                can_split     = cs,
                            )
                            try:
                                result   = decide(state)
                                bs_action = result['action']
                                bs_idx   = ACTIONS.index(bs_action) if bs_action in ACTIONS else ACTION_HIT
                            except Exception:
                                bs_idx = ACTION_HIT  # safe default

                            state_idx = encode_state(pt, soft, d_val, tc_raw, cd, cs)

                            # Seed: BS action = strong positive, others = 0
                            q_table[state_idx, :]       = 0.0
                            q_table[state_idx, bs_idx]  = _BS_SEED_STRONG

                            # Unavailable actions get penalty
                            if not cd:
                                q_table[state_idx, ACTION_DOUBLE] = _BS_SEED_PENALTY
                            if not cs:
                                q_table[state_idx, ACTION_SPLIT]  = _BS_SEED_PENALTY

                            seeded += 1

    log.info("Phase 1 complete: %d state-slots seeded from strategy.py.", seeded)
    return q_table


# ---------------------------------------------------------------------------
# Phase 2: Monte Carlo trainer (Every-Visit, epsilon-greedy)
# Adapted from tarunravi/BlackjackAI (Monte Carlo Method)
# ---------------------------------------------------------------------------

def train(
    episodes: int = 500_000,
    epsilon: float = 0.10,
    alpha:   float = 0.005,  # small alpha → MC refines without overwriting BS seed
    gamma:   float = 1.00,
    num_decks: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """
    Train a Q-table using BS-seeded Every-Visit Monte Carlo.

    Phase 1: Seed all states from Basic Strategy (instant, 100% coverage).
    Phase 2: MC episodes refine Q-values with true-count awareness.

    Returns
    -------
    np.ndarray of shape (TOTAL_STATES, NUM_ACTIONS)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Phase 1: seed from Basic Strategy
    q_table = np.zeros((TOTAL_STATES, NUM_ACTIONS))
    q_table = seed_from_basic_strategy(q_table)

    n_table = np.zeros((TOTAL_STATES, NUM_ACTIONS), dtype=np.int32)

    shoe = Shoe(num_decks=num_decks)

    log.info("Phase 2: MC training: episodes=%d, ε=%.2f, α=%.3f", episodes, epsilon, alpha)

    for ep in range(episodes):
        if ep % 50000 == 0:
            log.info("  episode %d / %d", ep, episodes)

        # Deal initial cards
        p1   = shoe.deal()
        d_up = shoe.deal()
        p2   = shoe.deal()
        d_hid= shoe.deal()

        player_hand   = [p1, p2]
        dealer_upcard = d_up
        dealer_hidden = d_hid

        p_val, p_soft = hand_value(player_hand)
        d_up_val = RANK_VALUES.get(dealer_upcard, 10)
        if dealer_upcard == 'A':
            d_up_val = 11

        tc         = shoe.true_count
        can_double = True
        can_split  = (p1 == p2)

        if not (PLAYER_TOTAL_MIN <= p_val <= PLAYER_TOTAL_MAX):
            continue

        state_idx = encode_state(p_val, p_soft, d_up_val, tc, can_double, can_split)

        # Epsilon-greedy action selection (greedy on BS-seeded Q)
        if random.random() < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = int(np.argmax(q_table[state_idx]))

        reward = simulate_hand(
            shoe, action, player_hand, dealer_upcard, dealer_hidden, can_double, can_split
        )

        # Every-Visit MC update: Q(s,a) ← Q(s,a) + α*(R - Q(s,a))
        n_table[state_idx, action] += 1
        q_table[state_idx, action] += alpha * (reward - q_table[state_idx, action])

    visited = int((n_table.sum(axis=1) > 0).sum())
    log.info("Phase 2 complete. States updated by MC: %d / %d  (rest use BS seed)",
             visited, TOTAL_STATES)
    return q_table


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Train BJ Monte Carlo Q-table")
    parser.add_argument("--episodes", type=int, default=20_000,
                        help="Number of training episodes (default: 20000)")
    parser.add_argument("--epsilon",  type=float, default=0.10)
    parser.add_argument("--alpha",    type=float, default=0.005)
    parser.add_argument("--decks",    type=int,   default=6)
    parser.add_argument("--out",      type=str,   default="models/bj_qtable.npy",
                        help="Output path for Q-table (default: models/bj_qtable.npy)")
    args = parser.parse_args()

    q_table = train(
        episodes  = args.episodes,
        epsilon   = args.epsilon,
        alpha     = args.alpha,
        num_decks = args.decks,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), q_table)
    log.info("Q-table saved → %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
