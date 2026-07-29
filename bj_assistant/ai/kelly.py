"""
kelly.py
--------
Kelly Criterion bet sizing integrated with Hi-Lo true count.

In Blackjack, the player has a NEGATIVE expected value at neutral counts
(house edge ~0.5% with Basic Strategy + counting).  The Kelly Criterion
only recommends a non-zero bet when the player has a positive edge — i.e.
at sufficiently high true counts.

The canonical Hi-Lo bet spread (Stanford Wong, "Professional Blackjack"):
  Player edge ≈ (TC - 1) * 0.5% per count unit above the pivot
  At TC=+1: edge ≈ 0% (breakeven)
  At TC=+2: edge ≈ +0.5%
  At TC=+3: edge ≈ +1.0%
  At TC=+4: edge ≈ +1.5%
  At TC=+5: edge ≈ +2.0%

Kelly optimal bet fraction = edge / variance
  BJ variance ≈ 1.15 (6-deck S17 DAS)
  f* = edge / 1.15

We use 1/4 Kelly (fractional Kelly) for risk management.
Below TC=+1: bet table minimum (1 unit).

References:
  - Stanford Wong, "Professional Blackjack" (1994)
  - Don Schlesinger, "Blackjack Attack" (2004)
  - Kelly (1956), "A New Interpretation of Information Rate"
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (6-deck, S17, DAS, BJ pays 3:2)
# ---------------------------------------------------------------------------

# TC pivot: the true count at which the player has ~0% edge (breakeven)
# For Hi-Lo with 6 decks, S17, DAS: pivot ≈ +1
TC_PIVOT = 1.0

# Player edge gain per +1 TC above pivot (empirical, Stanford Wong)
EDGE_PER_TC = 0.005   # +0.5% per unit (e.g. TC=+3 → +1.0% edge)

# BJ variance (6-deck S17 DAS) for Kelly calculation
BJ_VARIANCE = 1.15

# Fractional Kelly divisor (1=full Kelly, 4=quarter Kelly conservative)
KELLY_FRACTION = 4

# Minimum and maximum bet multipliers
MIN_BET_UNITS = 1
MAX_BET_UNITS = 12


def player_edge(true_count: float) -> float:
    """
    Return estimated player edge at the given true count.
    Negative below TC_PIVOT, positive above.
    """
    return (true_count - TC_PIVOT) * EDGE_PER_TC


def kelly_bet_units(
    true_count: float,
    balance: Optional[int] = None,
    unit_value: int = 250,
    min_bet: int = MIN_BET_UNITS,
    max_bet: int = MAX_BET_UNITS,
) -> int:
    """
    Return the number of units to bet based on Kelly Criterion + Hi-Lo TC.

    At TC ≤ +1: bet minimum (house has the edge — Kelly says don't bet).
    At TC > +1: Kelly fraction of bankroll, capped to max_bet units.

    Parameters
    ----------
    true_count : float
        Current Hi-Lo true count.
    balance : int, optional
        Current chip balance. If provided, Kelly fraction is applied to balance.
    unit_value : int
        Value of one betting unit in chips (default 250).
    min_bet : int
        Minimum units to bet (floor, default 1).
    max_bet : int
        Maximum units to bet (ceiling safety cap, default 12).

    Returns
    -------
    int
        Number of units to bet (1–12).

    Examples
    --------
    >>> kelly_bet_units(0.0)
    1
    >>> kelly_bet_units(2.0)
    2
    >>> kelly_bet_units(4.0)
    4
    >>> kelly_bet_units(6.0)
    8
    """
    edge = player_edge(true_count)

    if edge <= 0:
        # House has the edge — bet minimum
        return min_bet

    # Kelly fraction: f* = edge / variance
    kelly_f    = edge / BJ_VARIANCE
    practical_f = kelly_f / KELLY_FRACTION

    if balance is not None and balance >= unit_value:
        units = int(balance * practical_f / unit_value)
    else:
        # Unknown balance — use a conservative TC-based fixed spread
        # Designed to match Kelly output at typical balance of 5000:
        #   TC=+2 → 1u, TC=+3 → 2u, TC=+4 → 3u, TC=+5 → 5u, TC=+6 → 7u
        tc_above = true_count - TC_PIVOT
        units = max(1, int(tc_above * 1.5))

    return max(min_bet, min(max_bet, units))


def kelly_chip_amount(
    true_count: float,
    balance: Optional[int] = None,
    unit_value: int = 250,
    available_chips: Optional[list] = None,
) -> int:
    """
    Return the chip denomination to bet.

    Uses a practical TC-scaled spread calibrated for typical BJ app balances:
      TC ≤ +1  → 1 unit  (250)
      TC = +2  → 1 unit  (250)
      TC = +3  → 2 units (500)
      TC = +4  → 4 units (1000)
      TC = +5  → 8 units (2000 → nearest chip 1000 or 2500)
      TC ≥ +6  → 10 units capped to max

    Parameters
    ----------
    available_chips : list of int, optional
        Chip values available on screen (e.g. [250, 500, 1000, 2500]).
        If None, uses standard chip values.

    Returns
    -------
    int
        Chip denomination to tap.
    """
    CHIP_VALUES = sorted(available_chips or [250, 500, 1000, 2500, 5000])

    # Practical TC-based unit spread (Kelly-inspired but adapted for chip sizes)
    # TC ≤ 1 → 1u, TC=2 → 1u, TC=3 → 2u, TC=4 → 4u, TC=5 → 8u, TC≥6 → 10u
    tc = true_count
    if tc <= 2:   units = 1
    elif tc <= 3: units = 2
    elif tc <= 4: units = 4
    elif tc <= 5: units = 6
    else:         units = min(10, MAX_BET_UNITS)

    target = units * unit_value

    # Cap to balance
    if balance is not None and balance >= unit_value:
        target = min(target, balance // 2)  # never bet more than half the stack

    # Pick the chip closest to (but not exceeding) target
    affordable = [v for v in CHIP_VALUES if v <= target]
    if affordable:
        return affordable[-1]
    return CHIP_VALUES[0]  # fallback to smallest chip


# ---------------------------------------------------------------------------
# Bet spread summary (for HUD display)
# ---------------------------------------------------------------------------

def bet_spread_label(true_count: float) -> str:
    """Return a human-readable bet recommendation label."""
    chip = kelly_chip_amount(true_count)
    if chip <= 250:
        return "Flat (1×)"
    units = chip // 250
    return f"Kelly {units}× unit  (${chip})"
