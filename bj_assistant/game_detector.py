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
    # Pixel-measured on 1080×2340 live frame (debug-frame verified 2026-07-27):
    #   Dealer bubble centre ≈ (362, 284) on 720×1560 scaled → (540, 436) on 1080×2340
    #   Player bubble centre ≈ (362, 1113) on 720×1560 → (540, 1658) on 1080×2340
    DEALER_BUBBLE_CX  = 0.500
    DEALER_BUBBLE_CY  = 0.186   # 436 / 2340
    DEALER_BUBBLE_R   = 0.048   # ~52px radius

    PLAYER_BUBBLE_CX  = 0.500
    PLAYER_BUBBLE_CY  = 0.708   # 1658 / 2340
    PLAYER_BUBBLE_R   = 0.048

    # Card rank top-left corner crops — verified against debug-frame 2026-07-27
    # On 1080×2340: dealer Q visible at approx x=240, y=360  (card top-left corner)
    #   rank text is in top-left ~15% of card, card width≈240px, height≈320px
    #   so rank region ≈ x=240-310, y=360-420 → fracs: x=0.222, y=0.154, w=0.065, h=0.026
    DEALER_CARD_RANK_X = 0.222
    DEALER_CARD_RANK_Y = 0.154
    DEALER_CARD_RANK_W = 0.065
    DEALER_CARD_RANK_H = 0.028

    # Player cards visible at approx x=265, y=788 (9♠ top-left) on 1080×2340
    #   rank region ≈ x=265-335, y=788-848 → fracs: x=0.245, y=0.337, w=0.065, h=0.026
    PLAYER_CARD_RANK_X = 0.245
    PLAYER_CARD_RANK_Y = 0.337
    PLAYER_CARD_RANK_W = 0.065
    PLAYER_CARD_RANK_H = 0.028

    # Buttons row — verified on debug-frame: Stand/Double/Hit at y≈2050–2220
    BUTTON_ROW_Y_TOP    = 0.877   # ≈ 2052 / 2340
    BUTTON_ROW_Y_BOTTOM = 0.962   # ≈ 2251 / 2340

    # Colour ranges (HSV) for each action button type + minimum pixel count per button.
    # Min px calibrated from debug-frame 2026-07-27 on 1080×2340:
    #   Stand  21460px  Double 20928px  (large coloured buttons)
    #   Hit      675px  (small green arrow — low threshold)
    #   Split  10706px  when NOT present this came from chip 1K edge
    #             → raise Split min to 15000 to reject chip bleed
    BUTTON_COLOURS = {
        #         HSV_lo              HSV_hi          min_px
        "Stand":  ((0,   100, 80),  (12,  255, 255), 5000),   # red   large btn
        "Hit":    ((50,  60,  80),  (90,  255, 255),  400),   # green small arrow
        "Double": ((95,  80,  80),  (135, 255, 255), 5000),   # blue  large btn
        "Split":  ((12,  100, 80),  (28,  255, 255), 15000),  # orange — chip bleed <10706
    }

    # Chip betting row — chips appear in the BOTTOM row only.
    # Verified positions on 1080×2340: all chips at y > 2100 (= 0.897×h).
    # Clear/Deal buttons sit at y ≈ 1821 (= 0.778×h) — above chip row.
    # Starting the scan at 0.88×h excludes those buttons from chip detection.
    CHIP_ROW_Y_TOP    = 0.880   # was 0.750 — raised to exclude Clear/Deal blobs
    CHIP_ROW_Y_BOTTOM = 0.975
    # Real chip denominations in Vegas Blackjack app (left to right)
    CHIP_VALUES  = [250, 500, 1000, 2500, 5000]

    # Betting screen Clear/Deal buttons — appear when a bet chip is placed.
    # Verified from live OCR scan: Clear at y≈0.80, Deal at y≈0.80 (above chip row).
    #   Clear: x=0–50%,  y=78–83%  → centre ≈ (270, 1880) on 1080×2340
    #   Deal:  x=50–100%, y=78–83% → centre ≈ (810, 1880) on 1080×2340
    DARK_BTN_Y_TOP    = 0.770
    DARK_BTN_Y_BOTTOM = 0.840
    DARK_BTN_CLEAR_X  = 0.25   # centre x of Clear button
    DARK_BTN_DEAL_X   = 0.75   # centre x of Deal button
    DARK_BTN_CY_FRAC  = 0.805  # centre y of both buttons

    # Game state detection — result overlay text region
    # IMPORTANT: must NOT overlap the permanent felt text:
    #   "BLACKJACK PAYS 3 TO 2"  at y≈27%–30%
    #   "Dealer Must Stand Soft 17"  at y≈30%–33%
    # Result banners ("Dealer Wins", "You Win", etc.) appear at y≈18%–26%
    RESULT_REGION_X = 0.05
    RESULT_REGION_Y = 0.18   # was 0.38 — moved UP to avoid felt text
    RESULT_REGION_W = 0.90
    RESULT_REGION_H = 0.10   # tight band: only the result banner area

    # Deal / Play button — appears after result or on betting screen.
    # Observed on live frame at (870, 1994) = y≈0.852, x≈0.806
    # Search the full bottom 25% of screen for a standalone green blob
    DEAL_BTN_Y_TOP    = 0.800
    DEAL_BTN_Y_BOTTOM = 0.965
    DEAL_BTN_X_LEFT   = 0.0
    DEAL_BTN_X_RIGHT  = 1.0
    DEAL_BTN_COLOUR_LO = (40, 80, 80)
    DEAL_BTN_COLOUR_HI = (90, 255, 255)
    DEAL_BTN_MIN_PX    = 2000   # must be a real button, not a small green icon

    # Clear button — appears on betting screen to remove current bet
    # Usually a red/orange button near the chip area
    # We detect it by looking for a red blob in the lower quarter that is NOT Stand
    CLEAR_BTN_Y_TOP   = 0.820
    CLEAR_BTN_Y_BOTTOM= 0.880
    CLEAR_BTN_COLOUR_LO = (0, 100, 80)
    CLEAR_BTN_COLOUR_HI = (15, 255, 255)
    CLEAR_BTN_MIN_PX    = 3000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GameFrame:
    """Parsed state extracted from a single captured frame."""
    dealer_total: Optional[int]        = None   # from bubble OCR
    player_total: Optional[int]        = None   # from bubble OCR
    is_soft: bool                      = False  # ace in player hand
    dealer_upcard_rank: Optional[str]  = None   # OCR'd from top card
    player_card_ranks: List[str]       = field(default_factory=list)
    buttons: dict[str, Tuple[int,int]] = field(default_factory=dict)
    # buttons = {"Stand":(cx,cy), "Hit":(cx,cy), ...} during playing phase
    chips: dict[int, Tuple[int,int]]    = field(default_factory=dict)
    deal_btn: Optional[Tuple[int,int]]  = None  # Deal/Play button
    clear_btn: Optional[Tuple[int,int]] = None  # Clear-bet button
    # dark_btns: {"Clear":(cx,cy), "Deal":(cx,cy)} — dark-background buttons
    dark_btns: dict[str, Tuple[int,int]] = field(default_factory=dict)
    game_state: str                     = "unknown"
    # "betting" | "playing" | "result" | "unknown"
    frame_w: int                       = 720
    frame_h: int                       = 1560
    balance: Optional[int]             = None   # player chip balance from top-left OCR
    strip_text: str                     = ""     # raw OCR of button strip (lowercased)

    @property
    def bet_is_placed(self) -> bool:
        """
        True when a bet chip is already on the table.

        Authoritative signal: the action strip (y=0.877-0.962) OCR.
          - NO bet:  strip reads '250 500 1k 2.5k sk'  (chip denominations)
          - HAS bet: strip reads 'clear deal'           (action buttons)

        The Clear/Deal buttons at y≈0.80 are ALWAYS visible regardless of
        whether a bet is placed, so we cannot use that zone.
        """
        # strip_text comes from the action strip (y=0.877-0.962) in detect()
        if "clear" in self.strip_text or "deal" in self.strip_text:
            return True
        return False

    @property
    def effective_dealer_upcard(self) -> Optional[str]:
        """
        Return the best available dealer upcard identifier:
          1. dealer_upcard_rank if OCR succeeded (e.g. "J", "A", "6")
          2. str(dealer_total) if bubble OCR gave us the total (e.g. "6", "10")
          3. None if neither is available

        Mapping for dealer_total from bubble:
          1  → "A"  (Ace always shows as 1 in the bubble when alone)
          2–9 → str(t)
          10–21 → "10" (strategy table uses "10" for 10/J/Q/K)
        """
        if self.dealer_upcard_rank is not None:
            return self.dealer_upcard_rank
        if self.dealer_total is not None:
            t = self.dealer_total
            if t == 1:
                return "A"   # Ace bubble shows as 1
            if t >= 10:
                return "10"
            return str(t)
        return None

    @property
    def is_actionable(self) -> bool:
        """True when we have enough info to make a strategy decision."""
        return (
            self.effective_dealer_upcard is not None
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

        btn_strip = self._get_button_strip(frame, w, h)
        gf.strip_text = self._ocr_text(btn_strip).lower()
        gf.game_state = self._detect_game_state(frame, w, h)

        if gf.game_state in ("playing", "result"):
            gf.dealer_total       = self._read_bubble(frame, w, h, "dealer")
            gf.player_total       = self._read_bubble(frame, w, h, "player")
            gf.dealer_upcard_rank = self._read_card_rank(frame, w, h, "dealer")
            gf.player_card_ranks  = self._read_player_ranks(frame, w, h)
            gf.is_soft            = self._detect_soft(gf.player_card_ranks)

        if gf.game_state == "playing":
            gf.buttons = self._detect_buttons(frame, w, h)

        if gf.game_state in ("betting", "result"):
            gf.chips = self._detect_chips(frame, w, h)
            # When bet is placed, Clear/Deal are at known fixed positions
            # (verified from live frame OCR: y≈0.805h, Clear x≈0.25w, Deal x≈0.75w)
            if gf.bet_is_placed:
                gf.clear_btn = (int(Layout.DARK_BTN_CLEAR_X * w), int(Layout.DARK_BTN_CY_FRAC * h))
                gf.deal_btn  = (int(Layout.DARK_BTN_DEAL_X  * w), int(Layout.DARK_BTN_CY_FRAC * h))
            else:
                # No bet yet — detect the green Deal button (for result phase)
                gf.deal_btn = self._detect_deal_button(frame, w, h)

        # Always read balance (visible in all states at top-left)
        gf.balance = self._read_balance(frame, w, h)

        log.debug(
            "Frame: state=%s dealer=%s player=%s(%s) upcard=%s btns=%s chips=%s",
            gf.game_state, gf.dealer_total, gf.player_total,
            "soft" if gf.is_soft else "hard",
            gf.dealer_upcard_rank, list(gf.buttons.keys()),
            list(gf.chips.keys())
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
        # ── 1. Button strip OCR — checked FIRST to catch "clear"/"deal" ──
        # The "clear/deal" betting screen keeps the Hit button visible (green),
        # which would otherwise be misclassified as "playing". OCR is the only
        # reliable way to distinguish this screen.
        btn_strip = self._get_button_strip(frame, w, h)
        btn_text  = self._ocr_text(btn_strip).lower()
        log.debug("Button strip OCR: %r", btn_text)

        # "clear" or "deal" in strip = betting screen with a placed bet
        BETTING_WORDS = ("clear", "deal")
        if any(word in btn_text for word in BETTING_WORDS):
            return "betting"

        # ── 2. Result overlay ────────────────────────────────────────────
        rx = int(Layout.RESULT_REGION_X * w)
        ry = int(Layout.RESULT_REGION_Y * h)
        rw = int(Layout.RESULT_REGION_W * w)
        rh = int(Layout.RESULT_REGION_H * h)
        result_roi  = frame[ry:ry+rh, rx:rx+rw]
        result_text = self._ocr_text(result_roi).lower()
        RESULT_KEYWORDS = ("dealer wins", "player wins", "you win", "push", "bust",
                           "dealer busts", "you bust", "it's a tie", "blackjack")
        if any(kw in result_text for kw in RESULT_KEYWORDS):
            return "result"

        # ── 3. Button colour detection (playing indicator) ───────────────
        if self._colour_buttons_visible(frame, w, h):
            return "playing"

        # ── 4. Button text OCR fallback ──────────────────────────────────
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

        # Primary indicator: bright-green Hit button (≥3000px = playing)
        hit_green_lo = np.array([45, 150, 150])
        hit_green_hi = np.array([90, 255, 255])
        hit_px = int(np.count_nonzero(cv2.inRange(hsv, hit_green_lo, hit_green_hi)))
        if hit_px >= 3000:
            log.debug("Hit bright-green px=%d → playing", hit_px)
            return True

        # Fallback: Hit small arrow (≥400px at lower saturation threshold)
        _, _, min_px_hit = Layout.BUTTON_COLOURS["Hit"]
        lo_h, hi_h, _ = Layout.BUTTON_COLOURS["Hit"]
        any_hit_px = int(np.count_nonzero(cv2.inRange(hsv, np.array(lo_h), np.array(hi_h))))
        if any_hit_px >= min_px_hit:
            log.debug("Hit green (relaxed) px=%d → playing", any_hit_px)
            return True

        log.debug("Hit bright-green px=%d, relaxed px=%d → not playing", hit_px, any_hit_px)
        return False

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
        """
        Read the rank from the top-left corner of the top face-up card.

        Strategy: scan a horizontal band above (dealer) or below (player) the
        score bubble to find the leftmost white card region, then OCR its
        top-left corner where the rank character lives.
        """
        if role == "dealer":
            # Dealer zone: y=9%–35% covers both cards (bubble at 18.6%)
            scan_y1 = int(0.09 * h)
            scan_y2 = int(0.35 * h)
        else:
            # Player zone: y=32%–70% covers player cards (bubble at 70.8%)
            scan_y1 = int(0.32 * h)
            scan_y2 = int(0.70 * h)

        zone = frame[scan_y1:scan_y2, 0:w]
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        # Cards are bright white (>200) on the dark felt
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Collect individual card-like white regions:
        #   - area 8 000–150 000 (single card, not the merged pile)
        #   - not touching left edge (x==0 = back of hidden card)
        #   - not touching right/UI area (x > 88% = gear icon)
        candidates = []
        max_single_card = int(0.20 * w * (scan_y2 - scan_y1))  # no bigger than 20% of zone
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = cw / max(ch, 1)
            if area < 8000 or area > max_single_card:
                continue
            if not (0.3 < aspect < 3.0):
                continue
            if x == 0 or x > int(0.88 * w):
                continue
            candidates.append((x, y, cw, ch, area))

        if not candidates:
            log.debug("_read_card_rank(%s): no card candidate found in %d contours",
                      role, len(contours))
            return None

        # Dealer: face-up card is typically the rightmost (visible) → sort by x desc
        # Player: first card is leftmost → sort by x asc
        candidates.sort(key=lambda r: r[0], reverse=(role == "dealer"))
        cx, cy, ccw, cch, best_area = candidates[0]

        # Rank text is in the top-left corner of the card
        rank_w = max(25, int(ccw * 0.28))
        rank_h = max(20, int(cch * 0.22))
        roi = zone[cy:cy+rank_h, cx:cx+rank_w]
        if roi.size == 0:
            return None

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Invert: card is white with dark rank text → black on white for Tesseract
        _, thresh = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
        big = cv2.resize(thresh, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

        text = self._ocr_ranks(big).strip().upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        # Normalise common Tesseract misreads for card ranks
        rank_map = {
            "T": "10",  # T → 10 (ten)
            "1": "A",   # 1 → Ace
            "O": "0",   # O → 0 (but "10" stays "10")
            "I": "1",   # I → 1 → Ace via next pass
            "L": "1",
            "G": "9",   # G misread of 9
        }
        text = rank_map.get(text, text)
        # Second pass for I→1→A
        if text == "1":
            text = "A"
        valid = {"A","2","3","4","5","6","7","8","9","10","J","Q","K"}
        result = text if text in valid else None
        log.debug("_read_card_rank(%s): zone=%d-%d candidates=%d best=(%d,%d,%d,%d) text=%r → %s",
                  role, scan_y1, scan_y2, len(candidates), cx, cy, ccw, cch, text, result)
        return result

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
    # Balance OCR (top-left of screen)
    # ------------------------------------------------------------------

    def _read_balance(self, frame: np.ndarray, w: int, h: int) -> Optional[int]:
        """
        Read the player's chip balance from the top-left corner.
        Verified on 1080×2340: the number "1,507" sits at x≈0.12–0.30, y=3–10%.
        A coin/medal icon to the left causes OCR to prepend junk digits.
        Fix: scan only x=12%–32% to skip the icon.
        """
        x1, y1 = int(0.17 * w), int(0.030 * h)
        x2, y2 = int(0.32 * w), int(0.110 * h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        big = cv2.resize(roi, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,"
        text = self._tess.image_to_string(big, config=cfg).strip() if self._tess else ""
        candidates = []
        for tok in re.findall(r'\d[\d,]*', text):
            try:
                val = int(tok.replace(",", ""))
                if 250 <= val <= 9_999_999:
                    candidates.append(val)
            except ValueError:
                pass
        if not candidates:
            log.debug("_read_balance: no valid number in %r", text)
            return None
        # Prefer the first token — that's the actual balance number
        val = candidates[0]
        log.debug("_read_balance: %d (raw=%r)", val, text)
        return val

    # ------------------------------------------------------------------
    # Chip detection (betting phase)
    # ------------------------------------------------------------------

    def _detect_chips(
        self, frame: np.ndarray, w: int, h: int
    ) -> "dict[int, Tuple[int,int]]":
        """
        Detect betting chip row during the betting phase.

        Finds all gold/amber blobs in the chip strip, sorts them left-to-right,
        and assigns chip denominations by position order (5, 25, 100, 500, 1000).
        This is more robust than fixed x-positions since the chip layout may vary.
        """
        y1 = int(Layout.CHIP_ROW_Y_TOP    * h)
        y2 = int(Layout.CHIP_ROW_Y_BOTTOM * h)
        strip = frame[y1:y2, 0:w]
        hsv   = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        # Broad gold/amber range — covers all chip denominations
        gold_lo = np.array([10, 60, 60])
        gold_hi = np.array([45, 255, 255])
        gold_mask = cv2.inRange(hsv, gold_lo, gold_hi)

        # Close to merge nearby pixels into chip blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel)

        total_px = int(np.count_nonzero(gold_mask))
        log.debug("Chip detection: total gold px=%d in strip y=%d-%d", total_px, y1, y2)

        contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue
            cx_b = bx + bw // 2
            cy_b = y1 + by + bh // 2
            blobs.append((cx_b, cy_b, area))
            log.debug("  Gold blob at (%d,%d) area=%.0f", cx_b, cy_b, area)

        if not blobs:
            log.debug("No chip blobs found")
            return {}

        # Filter out oversized blobs (the selected/stacked chip in the center
        # can be 10× bigger than individual chips — skip it).
        # Individual chip blobs are typically area < 50000.
        chip_blobs = [(cx_b, cy_b, area) for cx_b, cy_b, area in blobs if area < 50000]

        if not chip_blobs:
            # Fallback: all blobs are large (no chips on screen in normal position)
            log.debug("No individual chip blobs found (all oversized)")
            return {}

        # Sort left-to-right and assign denominations by position order
        chip_blobs.sort(key=lambda b: b[0])
        chips: dict[int, Tuple[int, int]] = {}
        for i, (cx_b, cy_b, _) in enumerate(chip_blobs[:5]):
            value = Layout.CHIP_VALUES[i] if i < len(Layout.CHIP_VALUES) else Layout.CHIP_VALUES[-1]
            chips[value] = (cx_b, cy_b)
            log.debug("Chip %d assigned at (%d,%d)", value, cx_b, cy_b)

        return chips

    def _detect_dark_buttons(
        self, frame: np.ndarray, w: int, h: int
    ) -> "dict[str, Tuple[int,int]]":
        """
        Detect the Clear and Deal buttons by OCR-ing the strip where they appear.

        Verified layout (1080×2340):
          Clear: x=0–50%,   y=77–84%  → centre ≈ (270, 1880)
          Deal:  x=50–100%, y=77–84%  → centre ≈ (810, 1880)

        We use OCR to confirm 'clear' and 'deal' text, then return the
        hardcoded centres (much more reliable than Canny blob detection).
        """
        y1 = int(Layout.DARK_BTN_Y_TOP    * h)
        y2 = int(Layout.DARK_BTN_Y_BOTTOM * h)
        mid = w // 2
        cy  = int(Layout.DARK_BTN_CY_FRAC * h)

        left_strip  = frame[y1:y2, 0:mid]
        right_strip = frame[y1:y2, mid:w]

        left_text  = self._ocr_text(left_strip).lower()
        right_text = self._ocr_text(right_strip).lower()
        log.debug("_detect_dark_buttons: left=%r right=%r", left_text[:40], right_text[:40])

        result: dict[str, Tuple[int, int]] = {}
        if "clear" in left_text:
            result["Clear"] = (int(Layout.DARK_BTN_CLEAR_X * w), cy)
        if "deal" in right_text:
            result["Deal"]  = (int(Layout.DARK_BTN_DEAL_X  * w), cy)

        if result:
            log.debug("Dark btns found: %s", {k: v for k, v in result.items()})
        return result

    def _detect_deal_button(
        self, frame: np.ndarray, w: int, h: int
    ) -> "Optional[Tuple[int,int]]":
        """
        Detect the Deal/Play button — the green button that starts the next hand.
        Searches the full bottom quarter for the largest green blob that is
        NOT the Hit button (which is at x≈0.62, y≈0.908).
        Returns (cx, cy) or None.
        """
        y1 = int(Layout.DEAL_BTN_Y_TOP    * h)
        y2 = int(Layout.DEAL_BTN_Y_BOTTOM * h)
        roi  = frame[y1:y2, 0:w]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lo = np.array(Layout.DEAL_BTN_COLOUR_LO)
        hi = np.array(Layout.DEAL_BTN_COLOUR_HI)
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < Layout.DEAL_BTN_MIN_PX:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw // 2
            cy = y1 + by + bh // 2
            # Skip the Hit button area (x≈620-720, y≈0.88-0.93)
            hit_x_lo, hit_x_hi = int(0.55 * w), int(0.75 * w)
            hit_y_lo, hit_y_hi = int(0.87 * h), int(0.94 * h)
            if hit_x_lo <= cx <= hit_x_hi and hit_y_lo <= cy <= hit_y_hi:
                continue
            if best is None or area > best[2]:
                best = (cx, cy, area)

        if best:
            log.debug("Deal button at (%d,%d) area=%.0f", best[0], best[1], best[2])
            return (best[0], best[1])
        return None

    def _detect_clear_button(
        self, frame: np.ndarray, w: int, h: int
    ) -> "Optional[Tuple[int,int]]":
        """
        Detect the Clear/Undo bet button on the betting screen.
        In Vegas BJ this is typically a red button near the chip area.
        Returns (cx, cy) or None.
        """
        y1 = int(Layout.CLEAR_BTN_Y_TOP    * h)
        y2 = int(Layout.CLEAR_BTN_Y_BOTTOM * h)
        roi  = frame[y1:y2, 0:w]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lo = np.array(Layout.CLEAR_BTN_COLOUR_LO)
        hi = np.array(Layout.CLEAR_BTN_COLOUR_HI)
        mask = cv2.inRange(hsv, lo, hi)
        px = int(np.count_nonzero(mask))
        if px < Layout.CLEAR_BTN_MIN_PX:
            return None

        ys, xs = np.where(mask > 0)
        cx = int(np.mean(xs))
        cy = y1 + int(np.mean(ys))
        log.debug("Clear button at (%d,%d) px=%d", cx, cy, px)
        return (cx, cy)

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

        for name, (lo, hi, min_px) in Layout.BUTTON_COLOURS.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            total_px = int(np.count_nonzero(mask))
            # First gate: must have enough pixels of this colour
            if total_px < min_px:
                log.debug("Button %s: %d px < min %d → absent", name, total_px, min_px)
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cx = bx + bw // 2
                cy = y1 + by + bh // 2  # back to full-frame coords
                buttons[name] = (cx, cy)
                log.debug("Button %s at (%d,%d) area=%.0f", name, cx, cy, cv2.contourArea(cnt))
                break

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
