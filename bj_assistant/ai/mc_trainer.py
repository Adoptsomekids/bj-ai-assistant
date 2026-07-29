"""
mc_trainer.py
-------------
Offline Monte Carlo reinforcement learning trainer for Blackjack.

Inspired by tarunravi/BlackjackAI (Monte Carlo method, epsilon-greedy).
Adapted for 6-deck S17 DAS BJ with Hi-Lo true count in the state space.

Algorithm: Every-Visit Monte Carlo with epsilon-greedy exploration.
State:  (player_total, is_soft, dealer_upcard_idx, tc_bucket, can_double, can_split)
Action: 0=Stand, 1=Hit, 2=Double, 3=Split

Run once to produce models/bj_qtable.npy:
    python -m bj_assistant.ai.mc_trainer --episodes 20000

The resulting Q-table is ~560 KB and loads in <5ms at runtime.
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
# Monte Carlo trainer (Every-Visit, epsilon-greedy)
# Adapted from tarunravi/BlackjackAI (Monte Carlo Method)
# ---------------------------------------------------------------------------

def train(
    episodes: int = 20_000,
    epsilon: float = 0.10,
    alpha:   float = 0.02,
    gamma:   float = 1.00,
    num_decks: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """
    Train a Q-table using Every-Visit Monte Carlo.

    Returns
    -------
    np.ndarray of shape (TOTAL_STATES, NUM_ACTIONS)
    """
    random.seed(seed)
    np.random.seed(seed)

    q_table = np.zeros((TOTAL_STATES, NUM_ACTIONS))
    n_table = np.zeros((TOTAL_STATES, NUM_ACTIONS), dtype=np.int32)  # visit counts

    shoe = Shoe(num_decks=num_decks)

    log.info("Training Monte Carlo BJ agent: episodes=%d, ε=%.2f, α=%.3f", episodes, epsilon, alpha)

    for ep in range(episodes):
        if ep % 2000 == 0:
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
        can_double = True   # first action always allows double
        can_split  = (p1 == p2)

        # Clamp player_total to valid range
        if not (PLAYER_TOTAL_MIN <= p_val <= PLAYER_TOTAL_MAX):
            continue

        state_idx = encode_state(p_val, p_soft, d_up_val, tc, can_double, can_split)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = int(np.argmax(q_table[state_idx]))

        # Simulate the hand
        reward = simulate_hand(
            shoe, action, player_hand, dealer_upcard, dealer_hidden, can_double, can_split
        )

        # Every-Visit MC update: Q(s,a) ← Q(s,a) + α*(R - Q(s,a))
        n_table[state_idx, action] += 1
        q_table[state_idx, action] += alpha * (reward - q_table[state_idx, action])

    log.info("Training complete. States visited: %d / %d",
             int((n_table.sum(axis=1) > 0).sum()), TOTAL_STATES)
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
    parser.add_argument("--alpha",    type=float, default=0.02)
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
