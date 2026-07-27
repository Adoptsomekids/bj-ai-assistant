"""
engine.py
---------
Main control loop that ties together:
  - Screen capture (ADB / scrcpy / macOS)
  - Game-specific detector (Vegas BJ app — score bubbles + button colours)
  - Strategy decision (Basic Strategy + Hi-Lo counting + Illustrious 18)
  - HUD overlay update
  - Optional ADB tap automation

The engine runs in its own thread at a configurable FPS and exposes a simple
start/stop API so it can be embedded in a CLI or GUI application.
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Optional

import cv2

from .capture import ScreenCapture, get_best_capture, ADBCapture
from .game_detector import VegasBJDetector, GameFrame
from .strategy import GameState, HiLoCounter, decide, hand_total
from .overlay import HUDOverlay

log = logging.getLogger(__name__)


class BJEngine:
    """
    Orchestrates the full pipeline:
      capture → detect → decide → display (→ optionally tap)
    """

    def __init__(
        self,
        capture: Optional[ScreenCapture] = None,
        device_serial: Optional[str] = None,
        fps: int = 5,
        auto_tap: bool = False,
        show_overlay: bool = True,
        use_tkinter: bool = False,
        decks: int = 6,
    ) -> None:
        self._capture = capture or get_best_capture(device_serial)
        self._detector = VegasBJDetector()
        self._counter = HiLoCounter(decks=decks)
        self._overlay = HUDOverlay(use_tkinter=use_tkinter) if show_overlay else None
        self._fps = fps
        self._auto_tap = auto_tap
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_decision: Optional[dict] = None
        # Track last seen player+dealer state to avoid re-counting same cards
        self._last_player_total: Optional[int] = None
        self._last_dealer_total: Optional[int] = None
        self._hand_counted: bool = False

        # ADB tap controller (only useful when auto_tap=True)
        self._adb: Optional[ADBCapture] = (
            self._capture if isinstance(self._capture, ADBCapture) else None
        )
        # Live button map populated by game_detector each frame
        self._live_buttons: dict = {}
        # Auto-tap de-duplication state
        self._last_tap_time: float = 0.0
        self._last_tapped_hand: tuple = (-1, -1)
        # Betting phase state
        self._bet_placed: bool = False
        self._bet_phase_entered: float = 0.0   # monotonic time when betting phase started

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine loop and (if enabled) the overlay."""
        if self._running:
            return
        self._running = True
        if self._overlay:
            self._overlay.start_async()
            time.sleep(0.3)  # let overlay window initialise
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("BJ Engine started (fps=%d, auto_tap=%s)", self._fps, self._auto_tap)

    def stop(self) -> None:
        self._running = False
        if self._overlay:
            self._overlay.stop()
        if self._thread:
            self._thread.join(timeout=5)
        self._capture.release()
        log.info("BJ Engine stopped")

    def reset_count(self) -> None:
        """Call this at the start of a new shoe."""
        self._counter.reset()
        self._last_player_total = None
        self._last_dealer_total = None
        self._hand_counted = False
        log.info("Card count reset")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        interval = 1.0 / max(self._fps, 1)
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                log.error("Engine tick error: %s", exc, exc_info=True)
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, interval - elapsed)
            time.sleep(sleep_for)

    def _tick(self) -> None:
        frame = self._capture.grab()
        if frame is None:
            return

        # ── Parse the frame ──────────────────────────────────────────
        gf: GameFrame = self._detector.detect(frame)
        self._live_buttons = gf.buttons

        # ── Result phase: reset tracking + auto-tap Deal button ──────
        if gf.game_state == "result":
            if self._hand_counted:
                self._hand_counted = False
                self._last_player_total = None
                self._last_dealer_total = None
                self._bet_placed         = False
                self._bet_phase_entered  = 0.0   # ready for next betting phase
            # Auto-tap Deal/New-round button so next hand starts
            if self._auto_tap and self._adb and gf.deal_btn:
                now = time.monotonic()
                if now - self._last_tap_time > self._TAP_COOLDOWN:
                    x, y = gf.deal_btn
                    log.info("Auto-tap: Deal button at (%d,%d)", x, y)
                    self._adb.tap(x, y)
                    self._last_tap_time = now
            return

        # ── Betting phase: tap the right chip based on TC ────────────
        if gf.game_state == "betting":
            now = time.monotonic()
            # Track when we first entered this betting phase
            if self._bet_phase_entered == 0.0:
                self._bet_phase_entered = now
                self._bet_placed = False   # fresh hand, reset

            tc  = self._counter.true_count()
            rc  = self._counter.running_count
            bet = (1 if tc <= 1 else 2 if tc <= 2 else 4 if tc <= 3 else 8 if tc <= 4 else 12)
            if self._overlay:
                self._overlay.update({
                    "phase": "betting",
                    "true_count": round(tc, 2),
                    "running_count": rc,
                    "bet_units": bet,
                })
            if self._auto_tap and self._adb and not self._bet_placed:
                if gf.chips:
                    # Chips detected — place bet normally
                    self._place_bet(gf)
                elif now - self._bet_phase_entered > 8.0:
                    # Waited 8s with no chips — fallback: tap center of button row
                    # (in case chip detection failed but a bet chip is visible there)
                    cx = gf.frame_w // 2
                    cy = int(0.919 * gf.frame_h)   # center of btn strip
                    log.warning("Auto-bet: no chips detected after 8s — tapping center (%d,%d)", cx, cy)
                    self._adb.tap(cx, cy)
                    time.sleep(0.6)
                    if gf.deal_btn:
                        self._adb.tap(gf.deal_btn[0], gf.deal_btn[1])
                    self._bet_placed = True
                    self._bet_phase_entered = 0.0
            return

        if not gf.is_actionable:
            return  # not enough info yet

        # Use rank OCR result if available; fall back to bubble total as upcard
        dealer_upcard = gf.effective_dealer_upcard
        player_total  = gf.player_total
        is_soft       = gf.is_soft

        # ── Noise filter ─────────────────────────────────────────────
        # During card-deal animation the bubble briefly renders partial numbers
        # (e.g. "1", "2", "3").  Valid two-card starting totals are 4–21.
        # Totals outside that range are OCR noise — skip the frame.
        if player_total is None or not (4 <= player_total <= 21):
            log.debug("Skipping implausible player_total=%s (animation noise)", player_total)
            return

        # ── Hi-Lo count update ───────────────────────────────────────
        # Count once per unique hand (when totals change = new hand started)
        new_hand = (
            player_total != self._last_player_total
            or gf.dealer_total != self._last_dealer_total
        )
        if new_hand and not self._hand_counted:
            # Count all visible card ranks this frame
            for rank in gf.player_card_ranks:
                self._counter.update(rank)
            if dealer_upcard:
                self._counter.update(dealer_upcard)
            self._last_player_total = player_total
            self._last_dealer_total = gf.dealer_total
            self._hand_counted = True

        # ── Strategy decision ────────────────────────────────────────
        # Build player_cards for the strategy engine from the bubble total.
        # All components must be valid rank strings so hand_total() works.
        #
        # Soft totals  12-21 → ["A", str(total-11)]   e.g. soft 18 → ["A","7"]
        # Hard totals 12-21  → ["10", str(total-10)]  e.g. hard 16 → ["10","6"]
        # Hard totals  4-11  → ["2",  str(total-2)]   e.g. hard  6 → ["2","4"]
        #   (total-2 gives 2-9, all valid; total-10 would give negatives for <12)
        # Special: player_total==11 with no ranks is almost always a single Ace
        #   at deal-start → treat as soft so strategy uses the soft table.
        if gf.player_card_ranks:
            player_cards = gf.player_card_ranks
        elif is_soft and 12 <= player_total <= 21:
            player_cards = ["A", str(player_total - 11)]
        elif player_total == 11 and not is_soft:
            # Single Ace showing (soft 11) — rank OCR missed it
            player_cards = ["A"]
            is_soft = True
        elif player_total >= 12:
            player_cards = ["10", str(player_total - 10)]
        else:
            # Hard 4-11
            player_cards = ["2", str(player_total - 2)]

        state = GameState(
            player_cards=player_cards,
            dealer_upcard=dealer_upcard or "",
            counter=self._counter,
            # cap_* derived from which buttons are actually on screen
            can_double=("Double" in gf.buttons),
            can_split=("Split" in gf.buttons),
            can_surrender=("Surrender" in gf.buttons),
        )

        decision = decide(state)
        # Store bubble totals for display — never the synthetic card components
        decision["player_total"]   = player_total
        decision["is_soft"]        = is_soft
        decision["player_cards"]   = player_cards
        decision["dealer_upcard"]  = dealer_upcard
        # player_display: clean human-readable total shown in HUD
        #   "18s" for soft 18, "16" for hard 16, "6" for hard 6
        decision["player_display"] = f"{player_total}{'s' if is_soft else ''}"
        self._last_decision = decision

        log.info(
            "Player=%d%s  Dealer=%s  → %s  TC=%.1f  Bet=%dx  Btns=%s",
            player_total, "(soft)" if is_soft else "",
            dealer_upcard,
            decision["label"], decision["true_count"], decision["bet_units"],
            list(gf.buttons.keys())
        )

        if self._overlay:
            self._overlay.update(decision)

        if self._auto_tap and self._adb and gf.buttons:
            self._execute_action(decision, gf)

    # ------------------------------------------------------------------
    # Auto-tap (experimental)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bet placement (betting phase)
    # ------------------------------------------------------------------

    # Chip value → how many taps of that chip denomination to bet N units.
    # TC-based units: 1,2,4,8,12 → we tap the largest chip that fits.
    _CHIP_VALUES_DESC = [1000, 500, 100, 25, 5]

    def _place_bet(self, gf: GameFrame) -> None:
        """
        During the betting phase, tap chips to place the optimal bet.

        Strategy:
          1. Compute desired bet in units using the current true count.
          2. Convert units to a chip denomination (tap the single biggest chip
             that is ≤ desired amount, if multiple are needed tap up to 3 times).
          3. After placing, tap the Deal button to start the hand.
        """
        tc       = self._counter.true_count()
        # Same bet-sizing as strategy.py
        if tc <= 1:
            units = 1
        elif tc <= 2:
            units = 2
        elif tc <= 3:
            units = 4
        elif tc <= 4:
            units = 8
        else:
            units = 12

        if not gf.chips:
            log.debug("Betting phase but no chips detected — skipping bet tap")
            return

        # Pick best chip: largest denomination available that is ≤ units
        # Fall back to smallest chip if all chips are > units
        available = sorted(gf.chips.keys(), reverse=True)
        chosen = available[-1]  # default: smallest chip
        for v in available:
            if v <= units:
                chosen = v
                break

        coords = gf.chips.get(chosen)
        if not coords or not self._adb:
            return

        # Number of taps: how many times to tap this chip to reach ~units
        taps = max(1, min(3, round(units / chosen)))
        x, y = coords
        log.info("Auto-bet: TC=%.1f → %d units → chip %d × %d taps at (%d,%d)",
                 tc, units, chosen, taps, x, y)

        for _ in range(taps):
            self._adb.tap(x, y)
            time.sleep(0.3)

        self._bet_placed    = True
        self._last_tap_time = time.monotonic()

        # Tap Deal button to start the hand (with a short pause after betting)
        time.sleep(0.5)
        if gf.deal_btn and self._adb:
            dx, dy = gf.deal_btn
            log.info("Auto-tap: Deal button at (%d,%d)", dx, dy)
            self._adb.tap(dx, dy)
            self._last_tap_time = time.monotonic()

    # ------------------------------------------------------------------
    # Play action (playing phase)
    # ------------------------------------------------------------------

    # Mapping from strategy action codes to button names in the game UI.
    # "R" (Surrender) tries the Surrender button first; if not detected, falls to Hit.
    _ACTION_TO_BUTTON = {
        "H": "Hit",
        "S": "Stand",
        "D": "Double",
        "P": "Split",
        "R": "Surrender",   # will fall back to Hit if Surrender not on screen
    }

    # Seconds to wait after a tap before allowing another tap (prevents multi-tap same hand)
    _TAP_COOLDOWN = 2.5

    def _execute_action(self, decision: dict, gf: GameFrame) -> None:
        """Tap the correct button ONCE per unique hand, with cooldown."""
        now = time.monotonic()

        # De-duplicate: same hand total combo + cooldown → skip
        current_hand = (decision.get("player_total", 0), gf.dealer_total or 0)
        if (current_hand == self._last_tapped_hand and
                now - self._last_tap_time < self._TAP_COOLDOWN):
            return

        action   = decision.get("action", "H")
        btn_name = self._ACTION_TO_BUTTON.get(action, "Hit")

        # Prefer live button positions detected in this frame
        coords = gf.buttons.get(btn_name)

        # Fallback cascade for unavailable buttons:
        #   Surrender → Hit  (app has no Surrender button)
        #   Double    → Hit  (may not be available on 3+ cards)
        #   Split     → Hit  (may not be available)
        if coords is None:
            fallback = "Hit"
            if btn_name != "Hit":
                log.info("Auto-tap: '%s' not on screen → falling back to Hit", btn_name)
            coords   = gf.buttons.get(fallback)
            btn_name = fallback

        if coords and self._adb:
            x, y = coords
            log.info("Auto-tap: %s (%s) at (%d, %d)", action, btn_name, x, y)
            self._adb.tap(x, y)
            self._last_tap_time    = now
            self._last_tapped_hand = current_hand
        else:
            log.warning("Auto-tap: no tappable button found in frame (buttons=%s)",
                        list(gf.buttons.keys()))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def last_decision(self) -> Optional[dict]:
        return self._last_decision

    @property
    def true_count(self) -> float:
        return self._counter.true_count()
