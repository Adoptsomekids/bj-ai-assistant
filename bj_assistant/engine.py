"""
engine.py
---------
Main control loop that ties together:
  - Screen capture (ADB / scrcpy / macOS)
  - Card detection (template + OCR)
  - Strategy decision
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
from .card_detector import CardDetector, DetectedCard
from .strategy import GameState, HiLoCounter, decide
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
        decks: int = 6,
    ) -> None:
        self._capture = capture or get_best_capture(device_serial)
        self._detector = CardDetector()
        self._counter = HiLoCounter(decks=decks)
        self._overlay = HUDOverlay() if show_overlay else None
        self._fps = fps
        self._auto_tap = auto_tap
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_decision: Optional[dict] = None
        self._known_cards: set[str] = set()  # track already-counted card labels

        # ADB tap controller (only useful when auto_tap=True)
        self._adb: Optional[ADBCapture] = (
            self._capture if isinstance(self._capture, ADBCapture) else None
        )

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
        self._known_cards.clear()
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

        h, w = frame.shape[:2]
        cards = self._detector.detect(frame)
        cards = self._detector.assign_roles(cards, h)

        player_cards = [c.rank for c in cards if c.role == "player"]
        dealer_cards  = [c.rank for c in cards if c.role == "dealer"]

        if not player_cards or not dealer_cards:
            return  # nothing actionable yet

        dealer_upcard = dealer_cards[0]

        # Update counter with newly seen cards
        for card in cards:
            label = f"{card.role}:{card.label}:{card.bbox}"
            if label not in self._known_cards:
                self._counter.update(card.rank)
                self._known_cards.add(label)

        state = GameState(
            player_cards=player_cards,
            dealer_upcard=dealer_upcard,
            counter=self._counter,
        )

        decision = decide(state)
        decision["player_cards"] = player_cards
        decision["dealer_upcard"] = dealer_upcard
        self._last_decision = decision

        log.info(
            "Cards=%s  Dealer=%s  → %s  TC=%.1f  Bet=%dx",
            player_cards, dealer_upcard,
            decision["label"], decision["true_count"], decision["bet_units"]
        )

        if self._overlay:
            self._overlay.update(decision)

        if self._auto_tap and self._adb:
            self._execute_action(decision, frame)

    # ------------------------------------------------------------------
    # Auto-tap (experimental)
    # ------------------------------------------------------------------

    # Button layout map — pixel coordinates are game-specific and must be
    # calibrated via config/button_map.yaml before enabling auto-tap.
    DEFAULT_BUTTON_MAP = {
        "H": None,   # (x, y) of HIT button — fill in after calibration
        "S": None,   # STAND
        "D": None,   # DOUBLE
        "P": None,   # SPLIT
        "R": None,   # SURRENDER
    }

    def _execute_action(self, decision: dict, frame: object) -> None:
        action = decision.get("action", "H")
        coords = self.DEFAULT_BUTTON_MAP.get(action)
        if coords:
            x, y = coords
            log.info("Auto-tap: %s at (%d, %d)", action, x, y)
            self._adb.tap(x, y)
        else:
            log.debug("Auto-tap: no coordinates configured for action %s", action)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def last_decision(self) -> Optional[dict]:
        return self._last_decision

    @property
    def true_count(self) -> float:
        return self._counter.true_count()
