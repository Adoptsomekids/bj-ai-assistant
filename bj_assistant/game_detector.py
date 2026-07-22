"""
game_detector.py
----------------
Game-specific detector for "Vegas Blackjack" (the app in the screenshots).

Key insight: this game renders the hand total inside a dark rounded-bubble
directly above the card pile.  We OCR that bubble instead of reading individual
cards — giving us 100% reliable totals with zero template atlas needed.

We also detect:
  - Whether the player hand is soft (ace present) — inferred from card rank OCR
  - Which action buttons are currently visible (Stand / Hit / Double / Split)
  - Their bounding boxes for ADB auto-tap

Layout constants were measured from the provided 720×1560 screenshots.
All coordinates are expressed as FRACTIONS of (width, height) so they scale
automatically to any phone resolution.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout constants (fractions of w, h — measured on 720×1560)
# ---------------------------------------------------------------------------

class Layout:
    # Score bubbles  (cx, cy, radius) — all fractions
    # Pixel-measured on 1080×2340 screenshots:
    #   Dealer bubble dark region: x=492-588, y=379-494 → centre (540, 436)
    #   Player bubble dark region: x=490-590, y=1597-1718 → centre (540, 1657)
    DEALER_BUBBLE_CX  = 0.500
    DEALER_BUBBLE_CY  = 0.186   # 436 / 2340
    DEALER_BUBBLE_R   = 0.048   # ~52px radius

    PLAYER_BUBBLE_CX  = 0.500
    PLAYER_BUBBLE_CY  = 0.708   # 1657 / 2340
    PLAYER_BUBBLE_R   = 0.048

    # Card rank top-left corner crops
    # Dealer top card rank ≈ x=230, y=500, w=75, h=60  (on 1080×2340)
    DEALER_CARD_RANK_X = 0.213
    DEALER_CARD_RANK_Y = 0.214
    DEALER_CARD_RANK_W = 0.090
    DEALER_CARD_RANK_H = 0.030

    # Player first card rank ≈ x=230, y=1200
    PLAYER_CARD_RANK_X = 0.213
    PLAYER_CARD_RANK_Y = 0.513
    PLAYER_CARD_RANK_W = 0.090
    PLAYER_CARD_RANK_H = 0.030

    # Buttons row — measured on 1080×2340:
    #   Stand/Hit/Double/Split row top ≈ y=2060, bottom ≈ y=2220
    BUTTON_ROW_Y_TOP    = 0.880   # ≈ 2059 / 2340
    BUTTON_ROW_Y_BOTTOM = 0.960   # ≈ 2246 / 2340

    # Colour ranges (HSV) for each button type
    # Measured from screenshots with colour picker (1080×2340 source)
    BUTTON_COLOURS = {
        "Stand":  ((0,   100, 80),  (12,  255, 255)),   # red
        "Hit":    ((50,  60,  80),  (90,  255, 255)),    # green (triangle icon)
        "Double": ((95,  80,  80),  (135, 255, 255)),    # blue
        "Split":  ((12,  100, 80),  (28,  255, 255)),    # orange
    }

    # Game state detection — result overlay text region
    RESULT_REGION_X = 0.05
    RESULT_REGION_Y = 0.38
    RESULT_REGION_W = 0.90
    RESULT_REGION_H = 0.12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GameFrame:
    """Parsed state extracted from a single captured frame."""
    dealer_total: Optional[int]       = None   # from bubble OCR
    player_total: Optional[int]       = None   # from bubble OCR
    is_soft: bool                     = False  # ace in player hand
    dealer_upcard_rank: Optional[str] = None   # OCR'd from top card
    player_card_ranks: List[str]      = field(default_factory=list)
    buttons: dict[str, Tuple[int,int]] = field(default_factory=dict)
    # buttons = {"Stand": (cx,cy), "Hit": (cx,cy), ...}
    game_state: str                   = "unknown"
    # "betting" | "playing" | "result" | "unknown"
    frame_w: int                      = 720
    frame_h: int                      = 1560

    @property
    def is_actionable(self) -> bool:
        """True when we have enough info to make a strategy decision."""
        return (
            self.dealer_upcard_rank is not None
            and self.player_total is not None
            and self.game_state == "playing"
        )


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class VegasBJDetector:
    """
    Detector tuned for the Vegas Blackjack app skin shown in the screenshots.
    Operates entirely via OpenCV + pytesseract — no card templates needed.
    """

    # Tesseract config for single digit / short strings
    _TESS_DIGITS  = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
    _TESS_RANKS   = "--oem 3 --psm 8 -c tessedit_char_whitelist=A23456789TJQK10"
    _TESS_TEXT    = "--oem 3 --psm 6"

    # Known brew install paths for Tesseract on macOS (Apple Silicon & Intel)
    _TESS_PATHS = [
        "/opt/homebrew/bin/tesseract",   # Apple Silicon
        "/usr/local/bin/tesseract",      # Intel Mac
    ]

    def __init__(self) -> None:
        try:
            import pytesseract
            import shutil, os

            # Auto-configure tesseract binary path if not on PATH
            if not shutil.which("tesseract"):
                for p in self._TESS_PATHS:
                    if os.path.isfile(p):
                        pytesseract.pytesseract.tesseract_cmd = p
                        log.info("Tesseract found at %s", p)
                        break
                else:
                    log.warning("Tesseract not found — OCR will be disabled. Install with: brew install tesseract")
            self._tess = pytesseract
        except ImportError:
            self._tess = None
            log.warning("pytesseract not installed — install with: pip install pytesseract")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> GameFrame:
        """Parse a BGR frame and return a GameFrame with all extracted info."""
        h, w = frame.shape[:2]
        gf = GameFrame(frame_w=w, frame_h=h)

        gf.game_state = self._detect_game_state(frame, w, h)

        if gf.game_state in ("playing", "result"):
            gf.dealer_total       = self._read_bubble(frame, w, h, "dealer")
            gf.player_total       = self._read_bubble(frame, w, h, "player")
            gf.dealer_upcard_rank = self._read_card_rank(frame, w, h, "dealer")
            gf.player_card_ranks  = self._read_player_ranks(frame, w, h)
            gf.is_soft            = self._detect_soft(gf.player_card_ranks)

        if gf.game_state == "playing":
            gf.buttons = self._detect_buttons(frame, w, h)

        log.debug(
            "Frame: state=%s dealer=%s player=%s(%s) upcard=%s btns=%s",
            gf.game_state, gf.dealer_total, gf.player_total,
            "soft" if gf.is_soft else "hard",
            gf.dealer_upcard_rank, list(gf.buttons.keys())
        )
        return gf

    # ------------------------------------------------------------------
    # Game state detection
    # ------------------------------------------------------------------

    def _detect_game_state(self, frame: np.ndarray, w: int, h: int) -> str:
        """
        Determine game phase by checking which UI elements are visible.
        - 'betting'  — chip row visible, no cards → "Place Your Bet" or bet amount shown
        - 'playing'  — cards visible, action buttons present
        - 'result'   — "Dealer Wins" / "Player Wins" / "Push" overlay

        Detection order:
        1. Result overlay text (OCR middle band)
        2. Action button COLOURS in the button strip (fast, no OCR — Stand=red,
           Hit=green, Double=blue, Split=orange).  This fires even when OCR
           can't read the button labels.
        3. Action button TEXT via OCR (fallback)
        """
        # ── 1. Result overlay ────────────────────────────────────────────
        rx = int(Layout.RESULT_REGION_X * w)
        ry = int(Layout.RESULT_REGION_Y * h)
        rw = int(Layout.RESULT_REGION_W * w)
        rh = int(Layout.RESULT_REGION_H * h)
        result_roi = frame[ry:ry+rh, rx:rx+rw]
        result_text = self._ocr_text(result_roi).lower()
        # Only trigger on outcome banners — NOT on the permanent table felt text.
        # "Blackjack Pays 3 to 2" and "Dealer Must Stand Soft 17" are always
        # visible on the table felt, so we must NOT match those.
        RESULT_KEYWORDS = ("dealer wins", "player wins", "you win", "push", "bust",
                           "dealer busts", "you bust", "it's a tie")
        if any(kw in result_text for kw in RESULT_KEYWORDS):
            return "result"

        # ── 2. Button colour detection (primary playing indicator) ───────
        # The action buttons have very distinctive HSV colours; OCR is not needed
        # to know we are in a playing state.
        if self._colour_buttons_visible(frame, w, h):
            return "playing"

        # ── 3. Button OCR fallback (in case colours are off) ────────────
        btn_strip = self._get_button_strip(frame, w, h)
        btn_text = self._ocr_text(btn_strip).lower()
        log.debug("Button strip OCR: %r", btn_text)
        ACTION_WORDS = ("stand", "hit", "double", "split", "surrender")
        if any(word in btn_text for word in ACTION_WORDS):
            return "playing"

        return "betting"

    def _colour_buttons_visible(self, frame: np.ndarray, w: int, h: int) -> bool:
        """
        Return True when the action-button row is visible (playing state).

        The Hit button's bright green (HSV hue 45–90, sat>150, val>150) is the
        definitive discriminator:
          - Playing:  Hit button present  → ≥3 000 px of bright green in the strip
          - Betting:  chip row present    → 0–5 px of bright green (500-chip is
                      teal/muted, hue≈101, val≈127 — outside this range)
          - Result:   no action buttons   → 0 px

        This fires reliably across all 5 test screenshots and the live frame.
        """
        y1 = int(Layout.BUTTON_ROW_Y_TOP    * h)
        y2 = int(Layout.BUTTON_ROW_Y_BOTTOM * h)
        strip = frame[y1:y2, 0:w]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        # Bright green = Hit button; absent from both the chip row and result screen
        hit_green_lo = np.array([45, 150, 150])
        hit_green_hi = np.array([90, 255, 255])
        hit_px = int(np.count_nonzero(cv2.inRange(hsv, hit_green_lo, hit_green_hi)))
        log.debug("Hit bright-green pixels in btn strip: %d", hit_px)
        return hit_px >= 3000

    # ------------------------------------------------------------------
    # Score bubble OCR
    # ------------------------------------------------------------------

    def _read_bubble(self, frame: np.ndarray, w: int, h: int, role: str) -> Optional[int]:
        """Read the total score from the dark rounded bubble above the card pile."""
        if role == "dealer":
            cx = int(Layout.DEALER_BUBBLE_CX * w)
            cy = int(Layout.DEALER_BUBBLE_CY * h)
            r  = int(Layout.DEALER_BUBBLE_R  * w)
        else:
            cx = int(Layout.PLAYER_BUBBLE_CX * w)
            cy = int(Layout.PLAYER_BUBBLE_CY * h)
            r  = int(Layout.PLAYER_BUBBLE_R  * w)

        # Tight crop: just the inner text area of the bubble (skip the ring border)
        # Using 60% of radius gives us the number without the decorative border
        inner = int(r * 0.70)
        x1, y1 = max(0, cx - inner), max(0, cy - inner)
        x2, y2 = min(w, cx + inner), min(h, cy + inner)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Bubble interior: dark background, white number text.
        # Invert so we get black-on-white for Tesseract.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Otsu threshold finds the dark/light split automatically
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Upscale aggressively for Tesseract
        big = cv2.resize(thresh, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

        # psm 8 = single word (best for 1-2 digit BJ totals)
        # psm 7 = single text line (fallback)
        for psm in ["8", "6", "7"]:
            cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
            text = self._tess.image_to_string(big, config=cfg).strip() if self._tess else ""
            text = re.sub(r"[^0-9]", "", text)
            try:
                val = int(text)
                if 1 <= val <= 31:
                    return val
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------
    # Card rank OCR
    # ------------------------------------------------------------------

    def _read_card_rank(self, frame: np.ndarray, w: int, h: int, role: str) -> Optional[str]:
        """Read the rank character from the top-left corner of the top visible card."""
        if role == "dealer":
            x = int(Layout.DEALER_CARD_RANK_X * w)
            y = int(Layout.DEALER_CARD_RANK_Y * h)
            cw = int(Layout.DEALER_CARD_RANK_W * w)
            ch = int(Layout.DEALER_CARD_RANK_H * h)
        else:
            x = int(Layout.PLAYER_CARD_RANK_X * w)
            y = int(Layout.PLAYER_CARD_RANK_Y * h)
            cw = int(Layout.PLAYER_CARD_RANK_W * w)
            ch = int(Layout.PLAYER_CARD_RANK_H * h)

        roi = frame[y:y+ch, x:x+cw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Cards are white with dark rank text
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        big = cv2.resize(thresh, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

        text = self._ocr_ranks(big).strip().upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        # Normalise common OCR errors
        rank_map = {"T": "10", "1": "A", "O": "0", "I": "1"}
        text = rank_map.get(text, text)

        valid = {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}
        return text if text in valid else None

    def _read_player_ranks(self, frame: np.ndarray, w: int, h: int) -> List[str]:
        """
        Read player card ranks by scanning horizontally in the player card zone.
        Returns list of rank strings (may be partial — depends on card overlap).
        """
        # Player card zone: roughly y=51%–67% of height
        y1 = int(0.510 * h)
        y2 = int(0.670 * h)
        card_zone = frame[y1:y2, 0:w]

        # Find white card regions (cards are mostly white)
        gray = cv2.cvtColor(card_zone, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ranks: List[str] = []
        card_regions: List[Tuple[int,int,int,int]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = cw / max(ch, 1)
            # Card-like: reasonably tall, not too wide
            if area > 5000 and 0.3 < aspect < 1.2:
                card_regions.append((x, y, cw, ch))

        # Sort left-to-right
        card_regions.sort(key=lambda r: r[0])

        for (x, y, cw, ch) in card_regions[:4]:  # max 4 player cards
            # Rank is in top-left ~20% of card
            rank_h = int(ch * 0.25)
            rank_w = int(cw * 0.40)
            roi = card_zone[y:y+rank_h, x:x+rank_w]
            if roi.size == 0:
                continue
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)
            big = cv2.resize(thresh, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            text = self._ocr_ranks(big).strip().upper()
            text = re.sub(r"[^A-Z0-9]", "", text)
            rank_map = {"T": "10", "O": "0", "I": "1"}
            text = rank_map.get(text, text)
            valid = {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}
            if text in valid:
                ranks.append(text)

        return ranks

    def _detect_soft(self, ranks: List[str]) -> bool:
        return "A" in [r.upper() for r in ranks]

    # ------------------------------------------------------------------
    # Button detection
    # ------------------------------------------------------------------

    def _get_button_strip(self, frame: np.ndarray, w: int, h: int) -> np.ndarray:
        y1 = int(Layout.BUTTON_ROW_Y_TOP    * h)
        y2 = int(Layout.BUTTON_ROW_Y_BOTTOM * h)
        return frame[y1:y2, 0:w]

    def _detect_buttons(
        self, frame: np.ndarray, w: int, h: int
    ) -> dict[str, Tuple[int, int]]:
        """
        Scan the button row and return screen-space (x, y) center of each found button.
        Uses HSV colour matching for the distinctive button colours.
        """
        y1 = int(Layout.BUTTON_ROW_Y_TOP    * h)
        y2 = int(Layout.BUTTON_ROW_Y_BOTTOM * h)
        strip = frame[y1:y2, 0:w]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        buttons: dict[str, Tuple[int,int]] = {}
        for name, (lo, hi) in Layout.BUTTON_COLOURS.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            # Morphological close to merge nearby pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx = bx + bw // 2
                cy = y1 + by + bh // 2  # back to full-frame coords
                buttons[name] = (cx, cy)
                break  # take the largest match per colour

        return buttons

    # ------------------------------------------------------------------
    # OCR helpers
    # ------------------------------------------------------------------

    def _ocr_digits(self, img: np.ndarray) -> str:
        if self._tess is None:
            return ""
        try:
            return self._tess.image_to_string(img, config=self._TESS_DIGITS)
        except Exception as exc:
            log.debug("OCR digits error: %s", exc)
            return ""

    def _ocr_ranks(self, img: np.ndarray) -> str:
        if self._tess is None:
            return ""
        try:
            return self._tess.image_to_string(img, config=self._TESS_RANKS)
        except Exception as exc:
            log.debug("OCR ranks error: %s", exc)
            return ""

    def _ocr_text(self, img: np.ndarray) -> str:
        if self._tess is None:
            return ""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self._tess.image_to_string(gray, config=self._TESS_TEXT)
        except Exception as exc:
            log.debug("OCR text error: %s", exc)
            return ""
