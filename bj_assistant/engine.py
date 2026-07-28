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
        self._bet_phase_entered: float = 0.0
        # hands_completed: counts how many full hands have been seen.
        self._hands_completed: int = 0
        self._started_mid_hand: bool = False
        self._result_handled: bool = False   # True while still in result screen
        # Last bet denomination placed (to detect when we need to change it)
        self._last_bet_denomination: int = 0
        # Session stats
        self._wins: int = 0
        self._losses: int = 0
        self._pushes: int = 0

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

        # ── Result phase: count hand + reset + auto-tap Deal ─────────
        if gf.game_state == "result":
            # Parse win/loss/push text and update stats (once per result screen)
            if not self._result_handled:
                outcome = self._parse_result_text(frame)
                if outcome == "win":
                    self._wins += 1
                elif outcome == "loss":
                    self._losses += 1
                elif outcome == "push":
                    self._pushes += 1

            # Count this hand (whether we started mid-hand or played it fully)
            if not self._result_handled:
                self._hands_completed  += 1
                self._hand_counted      = False
                self._last_player_total = None
                self._last_dealer_total = None
                self._bet_placed        = False
                self._bet_phase_entered = 0.0
                self._started_mid_hand  = False
                self._result_handled    = True
                log.info("Hand #%d complete — W:%d L:%d P:%d",
                         self._hands_completed, self._wins, self._losses, self._pushes)

            # Auto-tap Deal to start next hand immediately
            if self._auto_tap and self._adb:
                if gf.deal_btn:
                    now = time.monotonic()
                    if now - self._last_tap_time > self._TAP_COOLDOWN:
                        log.info("Auto-tap: Deal at (%d,%d)", *gf.deal_btn)
                        self._adb.tap(*gf.deal_btn)
                        self._last_tap_time = now
            return

        # ── Betting phase: select chip + Deal ────────────────────────
        if gf.game_state == "betting":
            # Leaving result phase → clear result_handled so next result is counted
            self._result_handled = False

            now = time.monotonic()
            if self._bet_phase_entered == 0.0:
                self._bet_phase_entered = now
                self._bet_placed = False

            tc  = self._counter.true_count()
            rc  = self._counter.running_count
            bet = (1 if tc <= 1 else 2 if tc <= 2 else 4 if tc <= 3 else 8 if tc <= 4 else 12)
            if self._overlay:
                self._overlay.update({
                    "phase": "betting",
                    "true_count": round(tc, 2),
                    "running_count": rc,
                    "bet_units": bet,
                    "hands_completed": self._hands_completed,
                    "wins": self._wins,
                    "losses": self._losses,
                    "pushes": self._pushes,
                })
            # Auto-bet immediately (no first-hand gate)
            if self._auto_tap and self._adb and not self._bet_placed:
                if gf.chips:
                    self._place_bet(gf)
                elif gf.deal_btn:
                    # No chips but Deal button visible → a bet is already placed.
                    # Just tap Deal to start the hand.
                    now2 = time.monotonic()
                    if now2 - self._last_tap_time > self._TAP_COOLDOWN:
                        log.info("Auto-bet: no chips but Deal visible → tapping Deal at (%d,%d)", *gf.deal_btn)
                        self._adb.tap(*gf.deal_btn)
                        self._last_tap_time = now2
                        self._bet_placed = True
                        self._bet_phase_entered = 0.0
                elif now - self._bet_phase_entered > 8.0:
                    # Absolute fallback after 8s
                    cx = gf.frame_w // 2
                    cy = int(0.919 * gf.frame_h)
                    log.warning("Auto-bet fallback: tapping center (%d,%d)", cx, cy)
                    self._adb.tap(cx, cy)
                    time.sleep(0.6)
                    self._bet_placed = True
                    self._bet_phase_entered = 0.0
            return

        if not gf.is_actionable:
            # Clear result_handled whenever we're in a non-result state
            if gf.game_state == "playing":
                self._result_handled = False
            return  # not enough info yet

        # Clear result_handled on first actionable playing frame
        self._result_handled = False

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
        # Prefer rank OCR ranks IF their computed total matches the bubble total
        # (OCR can detect only 1 of 2 cards → partial ranks give wrong total).
        # If ranks are partial/wrong, fall back to synthetic two-card hand:
        #   Soft totals  12-21 → ["A", str(total-11)]   e.g. soft 18 → ["A","7"]
        #   Hard totals 12-21  → ["10", str(total-10)]  e.g. hard 16 → ["10","6"]
        #   Hard totals  4-11  → ["2",  str(total-2)]   e.g. hard  6 → ["2","4"]
        # Special: player_total==11 with no ranks → single Ace → treat as soft.
        _ranks_ok = False
        if gf.player_card_ranks:
            t_check, s_check = hand_total(gf.player_card_ranks)
            _ranks_ok = (t_check == player_total and s_check == is_soft)
            if not _ranks_ok:
                log.debug(
                    "Rank OCR total %d(%s) != bubble %d(%s) — ignoring partial ranks",
                    t_check, "soft" if s_check else "hard",
                    player_total, "soft" if is_soft else "hard"
                )
        if _ranks_ok:
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
        decision["player_total"]      = player_total
        decision["is_soft"]           = is_soft
        decision["player_cards"]      = player_cards
        decision["dealer_upcard"]     = dealer_upcard
        decision["player_display"]    = f"{player_total}{'s' if is_soft else ''}"
        # Session stats for HUD
        decision["wins"]              = self._wins
        decision["losses"]            = self._losses
        decision["pushes"]            = self._pushes
        decision["hands_completed"]   = self._hands_completed
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
    # Session stats
    # ------------------------------------------------------------------

    def _parse_result_text(self, frame) -> Optional[str]:
        """
        OCR the result banner and return 'win', 'loss', or 'push'.
        Returns None if result cannot be determined.
        """
        import re as _re
        h, w = frame.shape[:2]
        rx = int(0.05 * w)
        ry = int(0.10 * h)
        rw = int(0.90 * w)
        rh = int(0.18 * h)
        roi = frame[ry:ry+rh, rx:rx+rw]
        text = self._detector._ocr_text(roi).lower()
        log.debug("Result text: %r", text)
        WIN_WORDS  = ("player wins", "you win", "you won", "blackjack", "dealer busts",
                      "dealer bust")
        LOSS_WORDS = ("dealer wins", "you lose", "you lost", "bust", "you bust")
        PUSH_WORDS = ("push", "tie", "it's a tie", "stand-off")
        if any(w in text for w in WIN_WORDS):
            return "win"
        if any(w in text for w in LOSS_WORDS):
            return "loss"
        if any(w in text for w in PUSH_WORDS):
            return "push"
        return None

    # ------------------------------------------------------------------
    # Bet placement (betting phase)
    # ------------------------------------------------------------------

    def _target_bet_taps(self) -> tuple:
        """
        Return (chip_index, taps) based on current true count.

        Chip indices map to CHIP_VALUES = [250, 500, 1000, 2500, 5000].
        Bet sizing (1 unit = 1× base chip = 250):
          TC ≤ 1  →  1 tap of 250   (bet 250)
          TC ≤ 2  →  2 taps of 250  (bet 500)
          TC ≤ 3  →  1 tap of 1000  (bet 1000)
          TC ≤ 4  →  1 tap of 2500  (bet 2500)
          TC ≥ 5  →  1 tap of 5000  (bet 5000)
        """
        tc = self._counter.true_count()
        if tc <= 1:   return (0, 1)   # 250 × 1
        elif tc <= 2: return (0, 2)   # 250 × 2
        elif tc <= 3: return (1, 1)   # 500 × 1  (≈ 2 units)
        elif tc <= 4: return (2, 1)   # 1000 × 1 (≈ 4 units)
        else:         return (3, 1)   # 2500 × 1 (≈ 10 units)

    def _place_bet(self, gf: GameFrame) -> None:
        """
        Place the optimal bet based on the current true count.

        Flow:
          1. Pick chip index + taps from TC.
          2. Find that chip in gf.chips by denomination.
          3. Tap chip N times, then tap Deal.
        """
        if not self._adb or not gf.chips:
            log.debug("_place_bet: no adb or no chips detected")
            return

        chip_idx, taps = self._target_bet_taps()
        from .game_detector import Layout
        # Map chip index to denomination
        target_denom = Layout.CHIP_VALUES[chip_idx] if chip_idx < len(Layout.CHIP_VALUES) else Layout.CHIP_VALUES[0]

        # Find closest available chip denomination (in case some are missing)
        available = sorted(gf.chips.keys())
        chosen = min(available, key=lambda v: abs(v - target_denom))
        tc = self._counter.true_count()
        log.info("Auto-bet: TC=%.1f → chip %d × %d taps", tc, chosen, taps)

        x, y = gf.chips[chosen]
        for i in range(taps):
            self._adb.tap(x, y)
            time.sleep(0.4)

        self._last_bet_denomination = chosen
        self._bet_placed            = True
        self._last_tap_time         = time.monotonic()

        # Short pause then tap Deal
        time.sleep(0.6)
        if gf.deal_btn:
            log.info("Auto-tap: Deal at (%d,%d)", *gf.deal_btn)
            self._adb.tap(*gf.deal_btn)
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
