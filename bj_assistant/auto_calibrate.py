"""
auto_calibrate.py
-----------------
Automatic calibration tool for the Vegas BJ app.

Given a screenshot, it:
  1. Detects all button positions and prints their pixel coords
  2. Detects score bubble positions
  3. Writes a calibrated config/settings.yaml button_map section
  4. Optionally previews the detected regions by drawing them on the image

Usage (CLI):
    python -m bj_assistant.auto_calibrate path/to/screenshot.jpg

Usage (programmatic):
    from bj_assistant.auto_calibrate import calibrate_from_file
    result = calibrate_from_file("screenshot.jpg")
    print(result)
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .game_detector import VegasBJDetector, Layout, GameFrame

log = logging.getLogger(__name__)


def calibrate_from_file(
    image_path: str | Path,
    preview: bool = False,
    save_preview: bool = True,
) -> dict:
    """
    Load a screenshot, run detection, return calibration result dict.

    Returns:
        {
            "resolution":  [w, h],
            "buttons":     {"Stand": [x,y], "Hit": [x,y], ...},
            "dealer_bubble": [cx, cy],
            "player_bubble": [cx, cy],
            "game_state":  "playing" | "betting" | ...
        }
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    detector = VegasBJDetector()
    gf: GameFrame = detector.detect(img)

    # Compute absolute bubble centres for reference
    dealer_bx = int(Layout.DEALER_BUBBLE_CX * w)
    dealer_by = int(Layout.DEALER_BUBBLE_CY * h)
    player_bx = int(Layout.PLAYER_BUBBLE_CX * w)
    player_by = int(Layout.PLAYER_BUBBLE_CY * h)

    result = {
        "resolution":     [w, h],
        "game_state":     gf.game_state,
        "dealer_total":   gf.dealer_total,
        "player_total":   gf.player_total,
        "dealer_upcard":  gf.dealer_upcard_rank,
        "player_ranks":   gf.player_card_ranks,
        "is_soft":        gf.is_soft,
        "dealer_bubble":  [dealer_bx, dealer_by],
        "player_bubble":  [player_bx, player_by],
        "buttons":        {k: list(v) for k, v in gf.buttons.items()},
    }

    if preview or save_preview:
        annotated = _draw_annotations(img, gf, w, h,
                                       dealer_bx, dealer_by,
                                       player_bx, player_by)
        if save_preview:
            out_path = Path(image_path).with_suffix(".annotated.jpg")
            cv2.imwrite(str(out_path), annotated)
            result["preview_saved"] = str(out_path)
            log.info("Annotated preview saved: %s", out_path)
        if preview:
            cv2.imshow("BJ AI — Calibration", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return result


def _draw_annotations(
    img: np.ndarray,
    gf: GameFrame,
    w: int, h: int,
    dealer_bx: int, dealer_by: int,
    player_bx: int, player_by: int,
) -> np.ndarray:
    """Draw detection overlays on a copy of the frame for debugging."""
    out = img.copy()
    r = int(Layout.DEALER_BUBBLE_R * w)

    # Dealer bubble
    cv2.circle(out, (dealer_bx, dealer_by), r + 5, (0, 255, 255), 2)
    label = f"D:{gf.dealer_total or '?'}"
    cv2.putText(out, label, (dealer_bx - 25, dealer_by - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Player bubble
    cv2.circle(out, (player_bx, player_by), r + 5, (0, 255, 0), 2)
    label = f"P:{gf.player_total or '?'}{'s' if gf.is_soft else ''}"
    cv2.putText(out, label, (player_bx - 25, player_by - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Dealer upcard rank region
    dx = int(Layout.DEALER_CARD_RANK_X * w)
    dy = int(Layout.DEALER_CARD_RANK_Y * h)
    dw = int(Layout.DEALER_CARD_RANK_W * w)
    dh = int(Layout.DEALER_CARD_RANK_H * h)
    cv2.rectangle(out, (dx, dy), (dx+dw, dy+dh), (0, 200, 255), 2)
    cv2.putText(out, f"up:{gf.dealer_upcard_rank or '?'}", (dx, dy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Buttons
    colours_bgr = {
        "Stand":  (50,  50,  200),
        "Hit":    (50,  200, 50),
        "Double": (200, 100, 50),
        "Split":  (50,  180, 200),
    }
    for name, (bx, by) in gf.buttons.items():
        colour = colours_bgr.get(name, (255, 255, 255))
        cv2.circle(out, (bx, by), 20, colour, 3)
        cv2.putText(out, name, (bx - 30, by - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    # Game state banner
    cv2.putText(out, f"State: {gf.game_state.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return out


def calibrate_all_screenshots(directory: str | Path, preview: bool = False) -> None:
    """Run calibration on every .jpg/.png in a directory and print a summary."""
    directory = Path(directory)
    images = sorted(list(directory.glob("*.jpg")) + list(directory.glob("*.png")))
    if not images:
        print(f"No images found in {directory}")
        return

    print(f"\nCalibrating {len(images)} screenshots from {directory}\n{'='*60}")
    for img_path in images:
        print(f"\n▶ {img_path.name}")
        try:
            result = calibrate_from_file(img_path, preview=preview, save_preview=True)
            print(f"  Resolution : {result['resolution']}")
            print(f"  Game state : {result['game_state']}")
            print(f"  Dealer     : total={result['dealer_total']}  upcard={result['dealer_upcard']}")
            print(f"  Player     : total={result['player_total']}  soft={result['is_soft']}  ranks={result['player_ranks']}")
            print(f"  Buttons    : {result['buttons']}")
            if result.get("preview_saved"):
                print(f"  Preview    : {result['preview_saved']}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n{'='*60}")
    print("Calibration complete. Check the .annotated.jpg files to verify detection accuracy.")
    print("If button positions look off, adjust Layout constants in game_detector.py.")


def generate_button_map_yaml(results: list[dict]) -> str:
    """
    Given calibration results from multiple screenshots, produce a
    button_map YAML snippet for config/settings.yaml.
    """
    # Aggregate and average button positions across frames
    sums: dict[str, list] = {}
    counts: dict[str, int] = {}
    for r in results:
        for name, coords in r.get("buttons", {}).items():
            sums.setdefault(name, [0, 0])
            sums[name][0] += coords[0]
            sums[name][1] += coords[1]
            counts[name] = counts.get(name, 0) + 1

    lines = ["button_map:"]
    action_map = {"Stand": "S", "Hit": "H", "Double": "D", "Split": "P"}
    for name, code in action_map.items():
        if name in sums and counts[name] > 0:
            ax = int(sums[name][0] / counts[name])
            ay = int(sums[name][1] / counts[name])
            lines.append(f"  {code}: [{ax}, {ay}]   # {name}")
        else:
            lines.append(f"  {code}: null            # {name} — not detected")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m bj_assistant.auto_calibrate <image_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        calibrate_all_screenshots(target)
    else:
        result = calibrate_from_file(target, save_preview=True)
        print(json.dumps(result, indent=2))
