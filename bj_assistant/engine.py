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

        # ── Result phase: reset hand tracking so next hand counts fresh
        if gf.game_state == "result":
            if self._hand_counted:
                # Hand ended — mark so next playing phase triggers new count
                self._hand_counted = False
                self._last_player_total = None
                self._last_dealer_total = None
            return

        if not gf.is_actionable:
            return  # not enough info yet

        # Use rank OCR result if available; fall back to bubble total as upcard
        dealer_upcard = gf.effective_dealer_upcard
        player_total  = gf.player_total
        is_soft       = gf.is_soft

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
        # Build player_cards for the strategy engine.
        # If card-rank OCR gave us actual ranks, use those.
        # Otherwise build a minimal synthetic hand from the bubble total so that
        # hand_total() returns the correct value:
        #   - soft total → ["A", str(total - 11)]   e.g. soft 18 → ["A","7"]
        #   - hard total → ["10", str(total - 10)]  e.g. hard 16 → ["10","6"]
        #     clamped so both components are valid rank strings (2-10 or A)
        if gf.player_card_ranks:
            player_cards = gf.player_card_ranks
        elif is_soft and 12 <= player_total <= 21:
            player_cards = ["A", str(player_total - 11)]
        else:
            # Hard total: split into two components that hand_total() can parse
            low = max(2, player_total - 10)
            high = player_total - low
            player_cards = [str(high), str(low)]

        state = GameState(
            player_cards=player_cards,
            dealer_upcard=dealer_upcard,
            counter=self._counter,
        )

        decision = decide(state)
        # Override with direct bubble totals for display accuracy
        decision["player_total"] = player_total
        decision["is_soft"] = is_soft
        decision["player_cards"] = player_cards
        decision["dealer_upcard"] = dealer_upcard
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

    # Mapping from strategy action codes to button names in the game UI
    _ACTION_TO_BUTTON = {
        "H": "Hit",
        "S": "Stand",
        "D": "Double",
        "P": "Split",
        "R": "Stand",   # Surrender not in this app → fall back to Stand
    }

    def _execute_action(self, decision: dict, gf: GameFrame) -> None:
        action = decision.get("action", "H")
        btn_name = self._ACTION_TO_BUTTON.get(action, "Hit")

        # Prefer live button positions detected in this frame
        coords = gf.buttons.get(btn_name)

        # Fallback: if the desired button isn't visible (e.g. Split not available)
        # degrade gracefully: Double→Hit, Split→Hit
        if coords is None and btn_name in ("Double", "Split"):
            coords = gf.buttons.get("Hit")
            log.info("Auto-tap: %s not available → falling back to Hit", btn_name)
            btn_name = "Hit"

        if coords:
            x, y = coords
            log.info("Auto-tap: %s (%s) at (%d, %d)", action, btn_name, x, y)
            self._adb.tap(x, y)
        else:
            log.warning("Auto-tap: button '%s' not found in frame", btn_name)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def last_decision(self) -> Optional[dict]:
        return self._last_decision

    @property
    def true_count(self) -> float:
        return self._counter.true_count()
