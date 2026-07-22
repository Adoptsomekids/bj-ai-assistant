"""
Tests for strategy.py — Basic Strategy and Hi-Lo counting.
Run with: pytest tests/ -v
"""

import pytest
from bj_assistant.strategy import (
    hand_total, is_pair, decide, GameState, HiLoCounter, card_value
)


# ---------------------------------------------------------------------------
# hand_total
# ---------------------------------------------------------------------------

class TestHandTotal:
    def test_hard_hand(self):
        assert hand_total(["K", "7"]) == (17, False)

    def test_soft_hand(self):
        total, soft = hand_total(["A", "6"])
        assert total == 17
        assert soft is True

    def test_ace_reduction(self):
        total, soft = hand_total(["A", "A", "9"])
        assert total == 21
        assert soft is True

    def test_bust_prevention(self):
        total, soft = hand_total(["A", "A", "A", "9"])
        assert total == 12

    def test_blackjack(self):
        total, soft = hand_total(["A", "K"])
        assert total == 21
        assert soft is True

    def test_ten_value_cards(self):
        assert hand_total(["J", "Q"])[0] == 20
        assert hand_total(["10", "K"])[0] == 20


# ---------------------------------------------------------------------------
# is_pair
# ---------------------------------------------------------------------------

class TestIsPair:
    def test_pair_of_aces(self):
        assert is_pair(["A", "A"]) == "A"

    def test_pair_of_eights(self):
        assert is_pair(["8", "8"]) == "8"

    def test_not_pair(self):
        assert is_pair(["A", "K"]) is None

    def test_three_cards(self):
        assert is_pair(["5", "5", "5"]) is None


# ---------------------------------------------------------------------------
# HiLoCounter
# ---------------------------------------------------------------------------

class TestHiLoCounter:
    def test_count_low_cards(self):
        c = HiLoCounter(decks=6)
        for card in ["2", "3", "4", "5", "6"]:
            c.update(card)
        assert c.running_count == 5

    def test_count_high_cards(self):
        c = HiLoCounter(decks=6)
        for card in ["10", "J", "Q", "K", "A"]:
            c.update(card)
        assert c.running_count == -5

    def test_true_count(self):
        c = HiLoCounter(decks=2)
        c.running_count = 4
        c.cards_seen = 52  # 1 deck remaining
        assert abs(c.true_count() - 4.0) < 0.1

    def test_reset(self):
        c = HiLoCounter()
        c.update("A")
        c.reset()
        assert c.running_count == 0
        assert c.cards_seen == 0


# ---------------------------------------------------------------------------
# Basic Strategy decisions
# ---------------------------------------------------------------------------

class TestBasicStrategy:
    def _decide(self, player, dealer, tc=0.0):
        counter = HiLoCounter(decks=6)
        counter.running_count = int(tc * 3)  # approximate
        state = GameState(player_cards=player, dealer_upcard=dealer, counter=counter)
        return decide(state)

    def test_always_stand_on_20(self):
        result = self._decide(["K", "Q"], "6")
        assert result["action"] == "S"

    def test_always_split_aces(self):
        result = self._decide(["A", "A"], "5")
        assert result["action"] == "P"

    def test_always_split_eights(self):
        result = self._decide(["8", "8"], "10")
        assert result["action"] == "P"

    def test_hard_16_vs_10_stand(self):
        # With TC >= 0 deviation kicks in
        result = self._decide(["9", "7"], "10", tc=1)
        assert result["action"] == "S"

    def test_hard_11_double_vs_6(self):
        result = self._decide(["7", "4"], "6")
        assert result["action"] == "D"

    def test_soft_18_double_vs_6(self):
        result = self._decide(["A", "7"], "6")
        assert result["action"] == "D"

    def test_soft_18_stand_vs_7(self):
        result = self._decide(["A", "7"], "7")
        assert result["action"] == "S"

    def test_hard_12_stand_vs_4(self):
        result = self._decide(["7", "5"], "4")
        assert result["action"] == "S"

    def test_bet_units_increase_with_count(self):
        low  = self._decide(["K", "7"], "6", tc=0)
        high = self._decide(["K", "7"], "6", tc=5)
        assert high["bet_units"] > low["bet_units"]

    def test_result_contains_required_keys(self):
        result = self._decide(["5", "6"], "9")
        for key in ("action", "label", "reasoning", "true_count", "bet_units"):
            assert key in result
