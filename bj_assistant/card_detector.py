"""
card_detector.py
----------------
Detects playing cards in a screenshot (numpy BGR array) using:
  1. Template matching against a pre-built card template atlas (fast, preferred).
  2. OCR fallback via Tesseract for rank/suit text regions when templates miss.

The detector returns a list of DetectedCard objects with bounding boxes so the
overlay module can draw annotations directly on the mirrored screen.
"""

from __future__ import annotations
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "card_templates"

RANK_CHARS = set("A23456789JQK") | {"10"}
SUIT_CHARS = {"♠", "♥", "♦", "♣", "S", "H", "D", "C"}


@dataclass
class DetectedCard:
    rank: str          # e.g. "A", "K", "10"
    suit: str          # e.g. "S", "H", "D", "C"
    confidence: float  # 0.0 – 1.0
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in screen coords
    role: str = "unknown"  # "player" | "dealer" | "unknown"

    @property
    def label(self) -> str:
        return f"{self.rank}{self.suit}"


# ---------------------------------------------------------------------------
# Template-based detector
# ---------------------------------------------------------------------------

class TemplateCardDetector:
    """
    Matches each card template from the atlas against the frame.
    Templates should live in assets/card_templates/ named like AH.png, KS.png, 10D.png …
    """

    def __init__(self, threshold: float = 0.82) -> None:
        self.threshold = threshold
        self._templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not ASSETS_DIR.exists():
            log.warning("Card template directory not found: %s", ASSETS_DIR)
            return
        for p in ASSETS_DIR.glob("*.png"):
            name = p.stem.upper()  # e.g. "AH", "10D"
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._templates[name] = img
        log.info("Loaded %d card templates", len(self._templates))

    def detect(self, frame: np.ndarray) -> List[DetectedCard]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results: List[DetectedCard] = []

        for name, tpl in self._templates.items():
            h, w = tpl.shape
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= self.threshold)
            for pt in zip(*locs[::-1]):  # (x, y)
                rank = name[:-1] if len(name) > 1 else name[0]
                suit = name[-1]
                conf = float(res[pt[1], pt[0]])
                results.append(DetectedCard(
                    rank=rank, suit=suit, confidence=conf,
                    bbox=(pt[0], pt[1], w, h)
                ))

        return _nms(results)

    def is_ready(self) -> bool:
        return len(self._templates) > 0


# ---------------------------------------------------------------------------
# OCR-based detector (fallback)
# ---------------------------------------------------------------------------

class OCRCardDetector:
    """
    Uses Tesseract + contour detection to find card regions and read rank/suit.
    Slower than template matching but works with any card skin.
    """

    OCR_CONFIG = "--oem 3 --psm 10 -c tessedit_char_whitelist=A23456789TJQK"

    def __init__(self) -> None:
        try:
            import pytesseract
            self._tess = pytesseract
        except ImportError:
            self._tess = None
            log.warning("pytesseract not installed — OCR detector unavailable")

    def detect(self, frame: np.ndarray) -> List[DetectedCard]:
        if self._tess is None:
            return []

        cards: List[DetectedCard] = []
        regions = self._find_card_regions(frame)
        for (x, y, w, h) in regions:
            roi = frame[y:y+h, x:x+w]
            rank = self._ocr_rank(roi)
            if rank:
                cards.append(DetectedCard(
                    rank=rank, suit="?",
                    confidence=0.6,
                    bbox=(x, y, w, h)
                ))
        return cards

    def _find_card_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            area = w * h
            if 0.55 < aspect < 0.80 and 2000 < area < 60000:
                regions.append((x, y, w, h))
        return regions

    def _ocr_rank(self, roi: np.ndarray) -> Optional[str]:
        from PIL import Image
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        pil_img = Image.fromarray(thresh)
        try:
            text = self._tess.image_to_string(pil_img, config=self.OCR_CONFIG).strip().upper()
            text = re.sub(r"[^A-Z0-9]", "", text)
            if text in RANK_CHARS or text == "10":
                return text
            if text == "T":
                return "10"
        except Exception as exc:
            log.debug("OCR error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------

class CardDetector:
    """
    Uses template matching first; falls back to OCR if templates unavailable.
    """

    def __init__(self) -> None:
        self._template = TemplateCardDetector()
        self._ocr = OCRCardDetector()

    def detect(self, frame: np.ndarray) -> List[DetectedCard]:
        if self._template.is_ready():
            results = self._template.detect(frame)
            if results:
                return results
        log.debug("Template detector missed — falling back to OCR")
        return self._ocr.detect(frame)

    def assign_roles(
        self, cards: List[DetectedCard], frame_height: int
    ) -> List[DetectedCard]:
        """
        Heuristic: cards in the top 40% of the frame → dealer; bottom 60% → player.
        Works for most mobile BJ layouts.
        """
        split = frame_height * 0.40
        for card in cards:
            cy = card.bbox[1] + card.bbox[3] / 2
            card.role = "dealer" if cy < split else "player"
        return cards


# ---------------------------------------------------------------------------
# Non-maximum suppression helper
# ---------------------------------------------------------------------------

def _nms(cards: List[DetectedCard], iou_threshold: float = 0.3) -> List[DetectedCard]:
    if not cards:
        return []
    cards = sorted(cards, key=lambda c: -c.confidence)
    kept: List[DetectedCard] = []
    for cand in cards:
        overlap = any(_iou(cand.bbox, k.bbox) > iou_threshold for k in kept)
        if not overlap:
            kept.append(cand)
    return kept


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0
