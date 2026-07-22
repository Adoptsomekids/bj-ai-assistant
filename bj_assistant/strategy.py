"""
strategy.py
-----------
Implements the mathematically optimal Basic Strategy for BlackJack plus
a Hi-Lo card counting layer that adjusts decisions based on the true count.

References:
  - Wizard of Odds Basic Strategy (6-deck, S17, DAS)
  - Stanford Wong "Professional Blackjack" Hi-Lo system
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Card / Hand helpers
# ---------------------------------------------------------------------------

CARD_VALUES: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10, "A": 11,
}

HI_LO_COUNT: dict[str, int] = {
    "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


def card_value(card: str) -> int:
    return CARD_VALUES.get(card.upper(), 0)


def hand_total(cards: List[str]) -> Tuple[int, bool]:
    """Return (total, is_soft). Soft means an Ace is counted as 11."""
    total = 0
    aces = 0
    for c in cards:
        v = card_value(c)
        if c.upper() == "A":
            aces += 1
        total += v
    # reduce aces from 11 → 1 until not bust
    while total > 21 and aces:
        total -= 10
        aces -= 1
    is_soft = aces > 0 and total <= 21
    return total, is_soft


def is_pair(cards: List[str]) -> Optional[str]:
    """If exactly two cards of the same rank, return the rank, else None."""
    if len(cards) == 2:
        r0 = "10" if card_value(cards[0]) == 10 and cards[0].upper() not in ("A",) else cards[0].upper()
        r1 = "10" if card_value(cards[1]) == 10 and cards[1].upper() not in ("A",) else cards[1].upper()
        if r0 == r1:
            return r0
    return None


# ---------------------------------------------------------------------------
# Basic Strategy tables (6-deck, dealer stands on soft 17, DAS allowed)
# Encoding: H=Hit, S=Stand, D=Double, P=Split, R=Surrender(if allowed else H)
# ---------------------------------------------------------------------------

# Hard totals: rows = player total (5-21), cols = dealer upcard (2-A)
HARD_STRATEGY: dict[int, dict[str, str]] = {
    #        2     3     4     5     6     7     8     9    10     A
    5:  {"2":"H","3":"H","4":"H","5":"H","6":"H","7":"H","8":"H","9":"H","10":"H","A":"H"},
    6:  {"2":"H","3":"H","4":"H","5":"H","6":"H","7":"H","8":"H","9":"H","10":"H","A":"H"},
    7:  {"2":"H","3":"H","4":"H","5":"H","6":"H","7":"H","8":"H","9":"H","10":"H","A":"H"},
    8:  {"2":"H","3":"H","4":"H","5":"H","6":"H","7":"H","8":"H","9":"H","10":"H","A":"H"},
    9:  {"2":"H","3":"D","4":"D","5":"D","6":"D","7":"H","8":"H","9":"H","10":"H","A":"H"},
    10: {"2":"D","3":"D","4":"D","5":"D","6":"D","7":"D","8":"D","9":"D","10":"H","A":"H"},
    11: {"2":"D","3":"D","4":"D","5":"D","6":"D","7":"D","8":"D","9":"D","10":"D","A":"H"},
    12: {"2":"H","3":"H","4":"S","5":"S","6":"S","7":"H","8":"H","9":"H","10":"H","A":"H"},
    13: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"H","8":"H","9":"H","10":"H","A":"H"},
    14: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"H","8":"H","9":"H","10":"H","A":"H"},
    15: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"H","8":"H","9":"H","10":"R","A":"H"},
    16: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"H","8":"H","9":"R","10":"R","A":"R"},
    17: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
    18: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
    19: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
    20: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
    21: {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
}

# Soft totals: rows = non-ace card value (2-9), cols = dealer upcard
SOFT_STRATEGY: dict[int, dict[str, str]] = {
    #          2     3     4     5     6     7     8     9    10     A
    2:  {"2":"H","3":"H","4":"H","5":"D","6":"D","7":"H","8":"H","9":"H","10":"H","A":"H"},
    3:  {"2":"H","3":"H","4":"H","5":"D","6":"D","7":"H","8":"H","9":"H","10":"H","A":"H"},
    4:  {"2":"H","3":"H","4":"D","5":"D","6":"D","7":"H","8":"H","9":"H","10":"H","A":"H"},
    5:  {"2":"H","3":"H","4":"D","5":"D","6":"D","7":"H","8":"H","9":"H","10":"H","A":"H"},
    6:  {"2":"D","3":"D","4":"D","5":"D","6":"D","7":"H","8":"H","9":"H","10":"H","A":"H"},
    7:  {"2":"S","3":"D","4":"D","5":"D","6":"D","7":"S","8":"S","9":"H","10":"H","A":"H"},
    8:  {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
    9:  {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
}

# Pair splits: rows = pair rank, cols = dealer upcard
PAIR_STRATEGY: dict[str, dict[str, str]] = {
    #          2     3     4     5     6     7     8     9    10     A
    "2":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"P","8":"H","9":"H","10":"H","A":"H"},
    "3":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"P","8":"H","9":"H","10":"H","A":"H"},
    "4":  {"2":"H","3":"H","4":"H","5":"P","6":"P","7":"H","8":"H","9":"H","10":"H","A":"H"},
    "5":  {"2":"D","3":"D","4":"D","5":"D","6":"D","7":"D","8":"D","9":"D","10":"H","A":"H"},
    "6":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"H","8":"H","9":"H","10":"H","A":"H"},
    "7":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"P","8":"H","9":"H","10":"H","A":"H"},
    "8":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"P","8":"P","9":"P","10":"P","A":"P"},
    "9":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"S","8":"P","9":"P","10":"S","A":"S"},
    "10": {"2":"S","3":"S","4":"S","5":"S","6":"S","7":"S","8":"S","9":"S","10":"S","A":"S"},
    "A":  {"2":"P","3":"P","4":"P","5":"P","6":"P","7":"P","8":"P","9":"P","10":"P","A":"P"},
}

ACTION_LABELS = {
    "H": "HIT",
    "S": "STAND",
    "D": "DOUBLE DOWN",
    "P": "SPLIT",
    "R": "SURRENDER (or HIT if not available)",
}


# ---------------------------------------------------------------------------
# Hi-Lo Counter
# ---------------------------------------------------------------------------

@dataclass
class HiLoCounter:
    """Running and true Hi-Lo card count tracker."""
    decks: int = 6
    running_count: int = 0
    cards_seen: int = 0

    def update(self, card: str) -> None:
        self.running_count += HI_LO_COUNT.get(card.upper(), 0)
        self.cards_seen += 1

    def true_count(self) -> float:
        decks_remaining = max(self.decks - self.cards_seen / 52, 0.5)
        return self.running_count / decks_remaining

    def reset(self) -> None:
        self.running_count = 0
        self.cards_seen = 0


# ---------------------------------------------------------------------------
# Strategy Engine
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    player_cards: List[str] = field(default_factory=list)
    dealer_upcard: str = ""
    can_double: bool = True
    can_split: bool = True
    can_surrender: bool = True
    counter: HiLoCounter = field(default_factory=HiLoCounter)


def _normalise_upcard(card: str) -> str:
    v = card_value(card)
    if v == 10:
        return "10"
    if card.upper() == "A":
        return "A"
    return str(v)


def decide(state: GameState) -> dict:
    """
    Core decision function.
    Returns a dict with:
      - action:      short code (H/S/D/P/R)
      - label:       human-readable action
      - reasoning:   brief explanation
      - true_count:  current Hi-Lo true count
      - bet_units:   suggested bet units based on count (1-12 scale)
    """
    player = state.player_cards
    upcard = _normalise_upcard(state.dealer_upcard)
    total, is_soft = hand_total(player)
    pair_rank = is_pair(player)
    tc = state.counter.true_count()

    # --- Count-based bet sizing ---
    if tc <= 1:
        bet_units = 1
    elif tc <= 2:
        bet_units = 2
    elif tc <= 3:
        bet_units = 4
    elif tc <= 4:
        bet_units = 8
    else:
        bet_units = 12

    action = "H"
    reasoning = ""

    # 1. Pair split check
    if pair_rank and state.can_split:
        tbl = PAIR_STRATEGY.get(pair_rank, {})
        action = tbl.get(upcard, "H")
        reasoning = f"Pair of {pair_rank}s vs dealer {upcard} → Basic Strategy pair table"

    # 2. Soft total
    elif is_soft and total <= 21:
        non_ace = total - 11  # the non-ace card value in a soft hand
        # clamp to table range (2-9)
        key = max(2, min(9, non_ace))
        tbl = SOFT_STRATEGY.get(key, {})
        action = tbl.get(upcard, "H")
        reasoning = f"Soft {total} (A+{non_ace}) vs dealer {upcard} → Soft strategy table"

    # 3. Hard total
    else:
        key = max(5, min(21, total))
        tbl = HARD_STRATEGY.get(key, {})
        action = tbl.get(upcard, "S" if total >= 17 else "H")
        reasoning = f"Hard {total} vs dealer {upcard} → Hard strategy table"

    # Apply Hi-Lo deviations (Illustrious 18 subset)
    action, deviation_note = _apply_count_deviations(
        action, total, is_soft, upcard, tc, is_pair_hand=bool(pair_rank)
    )
    if deviation_note:
        reasoning += f" | Count deviation (TC={tc:.1f}): {deviation_note}"

    # If action needs capability check
    if action == "D" and not state.can_double:
        action = "H"
        reasoning += " (Double not available → Hit)"
    if action == "P" and not state.can_split:
        action = "H"
        reasoning += " (Split not available → Hit)"
    if action == "R" and not state.can_surrender:
        action = "H"
        reasoning += " (Surrender not available → Hit)"

    return {
        "action": action,
        "label": ACTION_LABELS.get(action, action),
        "reasoning": reasoning,
        "true_count": round(tc, 2),
        "running_count": state.counter.running_count,
        "bet_units": bet_units,
        "player_total": total,
        "is_soft": is_soft,
    }


def _apply_count_deviations(
    base_action: str,
    total: int,
    is_soft: bool,
    upcard: str,
    tc: float,
    is_pair_hand: bool = False,
) -> Tuple[str, str]:
    """
    Apply the most important Hi-Lo index deviations (Illustrious 18).
    Returns (possibly_new_action, deviation_note).
    """
    note = ""
    action = base_action
    # Never override a pair Split decision with a count deviation
    if is_pair_hand and base_action == "P":
        return action, note

    deviations = [
        # (player_total, is_soft, dealer_upcard, tc_threshold, new_action, description)
        (16, False, "9",  5, "S",  "Stand 16 vs 9 at TC≥5"),
        (16, False, "10", 0, "S",  "Stand 16 vs 10 at TC≥0"),
        (15, False, "10", 4, "S",  "Stand 15 vs 10 at TC≥4"),
        (13, False, "2",  -1,"H",  "Hit 13 vs 2 at TC≤-1"),
        (12, False, "2",  3, "S",  "Stand 12 vs 2 at TC≥3"),
        (12, False, "3",  2, "S",  "Stand 12 vs 3 at TC≥2"),
        (12, False, "4",  0, "S",  "Stand 12 vs 4 at TC≥0"),
        (11, False, "A",  1, "D",  "Double 11 vs A at TC≥1"),
        (10, False, "10", 4, "D",  "Double 10 vs 10 at TC≥4"),
        (10, False, "A",  4, "D",  "Double 10 vs A at TC≥4"),
        (9,  False, "2",  1, "D",  "Double 9 vs 2 at TC≥1"),
        (9,  False, "7",  3, "D",  "Double 9 vs 7 at TC≥3"),
        (20, False, "6",  5, "P",  "Split 10s vs 6 at TC≥5"),
    ]

    for p_total, p_soft, d_up, threshold, new_act, desc in deviations:
        if total == p_total and is_soft == p_soft and upcard == d_up:
            if (threshold >= 0 and tc >= threshold) or (threshold < 0 and tc <= threshold):
                action = new_act
                note = desc
                break

    return action, note
