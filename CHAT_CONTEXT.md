# BJ AI Assistant — Chat Context Backup

> **Date:** 2026-07-22 (updated after live test session)
> **GitHub repo:** https://github.com/Adoptsomekids/bj-ai-assistant
> **Local clone:** `/Users/emilio-ibm/Documents/MOD/BOB/BJ/bj-ai-assistant`
> **Language:** Python 3.9 (venv at `.venv/`)
> **Owner:** Adoptsomekids (GitHub account)

---

## ⚠️ Bob crash prevention rules (IMPORTANT)

1. **NEVER display images inline** — crashes Bob immediately.
2. **Keep responses SHORT** — one `apply_diff` at a time, no long code walls.
3. **Read files before editing** — always `read_file` first.
4. **Do not crash on `apply_diff`** — if context is large, read the file, then apply a minimal targeted diff.

---

## Project Goal

Build an intelligent real-time BlackJack AI assistant that:

1. **Mirrors the Android phone screen** to the Mac via USB (ADB `screencap`)
2. **Detects the game state** from each captured frame (betting / playing / result)
3. **Reads hand totals** from the score-bubble OCR
4. **Computes the optimal action** using Basic Strategy (6-deck S17 DAS) + Hi-Lo + Illustrious 18
5. **Shows a live terminal HUD** (Rich panel)
6. **Auto-taps the phone** — betting phase (chip by TC), playing phase (Hit/Stand/Double/Split), result phase (Deal)

### Full automation pipeline
```
1. User plays the FIRST hand manually (assistant only advises)
2. From hand #2 onwards, assistant plays ALONE:
   betting phase  → Clear if denomination changes → tap chip by TC → tap Deal
   playing phase  → tap optimal action (Basic Strategy + Hi-Lo + I18)
   result phase   → tap Deal for next hand
3. Repeats infinitely until Ctrl+C
```
Bet sizing by TC: ≤1→1×  ≤2→2×  ≤3→4×  ≤4→8×  ≥5→12×

---

## Target App

**"Vegas Blackjack"** (Android)
- Rules: **Dealer Stands Soft 17, Blackjack pays 3:2**
- Screen resolution: **1080 × 2340 px** (portrait)
- Score bubbles: dark rounded speech-bubble above each card pile — OCR'd for totals
- Chip values: **250, 500, 1K, 2.5K, 5K** (NOT 5/25/100/500/1000)
- Deal button (green): at approx (870, 1994) on 1080×2340

---

## Repo Structure

```
bj-ai-assistant/
├── bj_assistant/
│   ├── __init__.py
│   ├── capture.py          # ADB screen capture (auto-finds /opt/homebrew/bin/adb)
│   ├── card_detector.py    # Legacy (not used in main flow)
│   ├── game_detector.py    # ★ Vegas BJ detector (score bubbles + button colours)
│   ├── auto_calibrate.py   # Screenshots folder → annotated debug images
│   ├── strategy.py         # Full Basic Strategy + HiLoCounter + Illustrious 18
│   ├── overlay.py          # TerminalHUD (Rich) + TkinterHUD (optional)
│   ├── engine.py           # ★ Main loop: capture → detect → decide → display → tap
│   ├── config.py           # YAML settings loader
│   └── cli.py              # Click CLI: run / decide-cmd / count / debug-frame
├── assets/card_templates/  # Sample screenshots (5 JPGs)
├── config/settings.yaml
├── tests/
│   ├── test_strategy.py    # 29 unit tests — ALL PASS
│   └── test_capture.py
├── pyproject.toml
├── setup.py
├── requirements.txt
└── README.md
```

---

## Layout Constants (game_detector.py — 1080×2340)

| Region | cx/cy fraction | Notes |
|---|---|---|
| Dealer score bubble | cx=0.500, cy=0.186 | Dark speech bubble, shows dealer total |
| Player score bubble | cx=0.500, cy=0.708 | Dark speech bubble, shows player total |
| Bubble OCR radius | 0.048 × width ≈ 52px | Inner crop = 0.70× radius |
| Dealer upcard rank | x=0.213, y=0.214 | Top-left corner of top dealer card |
| Player card rank | x=0.213, y=0.513 | Top-left corner of top player card |
| Button row | y: 0.880–0.960 | Stand / Hit / Double / Split |
| Deal button | y: 0.80–0.965, full width | Detected in extended zone |
| Clear button | y: 0.82–0.88 | Betting phase |

---

## Button Color Detection (HSV ranges)

| Button | HSV Low | HSV High | Color |
|---|---|---|---|
| Stand | (0,100,80) | (12,255,255) | Red |
| Hit | (50,60,80) | (90,255,255) | Green |
| Double | (95,80,80) | (135,255,255) | Blue |
| Split | (12,100,80) | (28,255,255) | Orange |

State detection order (current, working):
1. Strip OCR FIRST: if `'clear'` or `'deal'` → `betting`
2. `_colour_buttons_visible()`: if ≥2 HSV colours detected → `playing`
3. Result overlay OCR: "dealer wins"/"player wins"/"push"/"bust" → `result`
4. Strip OCR action words → `playing`
5. Default → `betting`

---

## Detector Accuracy (5 test screenshots — ALL PASS ✅)

| Screenshot | State | Dealer | Player | Buttons |
|---|---|---|---|---|
| 015520 (2+J vs 10♦) | playing ✅ | 10 ✅ | 12 ✅ | Stand/Hit/Double/Split ✅ |
| 015611 (J+J vs 2♠) | playing ✅ | 2 ✅ | 20 ✅ | Stand/Hit/Double/Split ✅ |
| 015602 (Dealer Wins) | result ✅ | 21 ✅ | 20 ✅ | — ✅ |
| 015503 (Place Bet) | betting ✅ | — ✅ | — ✅ | — ✅ |
| 015512 (Deal screen) | betting ✅ | — ✅ | — ✅ | — ✅ |

---

## Strategy Engine

- **Tables:** Hard 5–21, Soft A+2 through A+9, Pairs 2–A (6-deck S17 DAS)
- **Hi-Lo counting:** running count + true count (running ÷ decks remaining)
- **Bet sizing:** 1× (TC≤1) → 2× (TC≤2) → 4× (TC≤3) → 8× (TC≤4) → 12× (TC≥5)
- **Illustrious 18 deviations:** Stand 16v10 at TC≥0, Double 11vA at TC≥+1, etc.
- **Actions:** H=Hit, S=Stand, D=Double, P=Split, R=Surrender(→Hit fallback)

---

## Setup & Run

```bash
# Prerequisites (already installed)
brew install android-platform-tools tesseract scrcpy

# Project
cd ~/Documents/MOD/BOB/BJ/bj-ai-assistant
source .venv/bin/activate     # Python 3.9 venv

# Verify phone connected
adb devices   # should show RZCW82D69YH  device

# Run live assistant (terminal HUD)
bj-assistant run

# With auto-tap
bj-assistant run --auto-tap

# With verbose debug logging
bj-assistant run -v

# Debug one frame (saves PNG to /tmp/, prints text table)
bj-assistant -v debug-frame

# One-shot decision (no phone needed)
bj-assistant decide-cmd --player "A 6" --dealer 5
```

---

## engine.py — Key Logic (current)

```python
_tick():
  1. gf = detector.detect(frame)
  2. if game_state == "result":
       if started_mid_hand and not hand_counted → hands_completed += 1
       if hand_counted → hands_completed += 1, reset state
       if auto_tap and hands_completed >= 1 → tap Deal
  3. if game_state == "betting":
       update HUD (TC, bet units)
       if auto_tap and hands_completed >= 1 → _place_bet(gf)
  4. if not gf.is_actionable → return
  5. noise filter: player_total not in 4-21 → return
  6. hi-lo count once per new hand
  7. build player_cards:
       rank OCR available  → use directly
       soft 12-21          → ["A", str(total-11)]
       total==11, not soft → ["A"]  (single Ace, treat as soft)
       hard 12-21          → ["10", str(total-10)]
       hard 4-11           → ["2",  str(total-2)]
  8. decide(state) → decision dict
  9. decision["player_display"] = f"{total}{'s' if soft else ''}"
 10. overlay.update(decision)
 11. if auto_tap → _execute_action(decision, gf)

overlay.py:
  - reads player_display (e.g. "6", "18s", "11s")
  - NEVER shows raw synthetic card components
```

---

## HUD Display Format

```
╭──────── ♠ BJ AI Assistant ─────────╮
│   🎯 HIT    You: 11   │ Dealer: 7  │
│   True Count: +0.2  Running: +1    │
│   Bet: 1× unit                     │
│   Hard 11 vs dealer 7 → Hit        │
╰────────────────────────────────────╯
```
- `You: 16` = hard 16
- `You: 18s` = soft 18 (Ace + 7)
- `You: 11s` = single Ace showing (soft 11)

---

## Known Issues / Bugs / Fixes

| Issue | Root Cause | Fix Applied | Status |
|---|---|---|---|
| `You: -1 2` in HUD | OCR reads `player_total=1` during deal animation; synthetic hand: `high=1-10=-1` | Noise filter: reject `player_total < 4` | ✅ Fixed |
| `You: 4 2` in HUD | HUD showed raw synthetic card components, not bubble total | Added `player_display` key; overlay uses it | ✅ Fixed |
| Hard 4-11 synthetic had negative rank | `make_cards(total)` for total<12: `high=total-10` goes negative | Use `["2", str(total-2)]` for hard 4-11 | ✅ Fixed |
| `You: As` (single Ace) | `player_total=11, is_soft=False` → built wrong hard-11 hand | Special-case: `total==11, not soft → ["A"], is_soft=True` | ✅ Fixed |
| `You: 2` in HUD | OCR reads `player_total=2` mid-animation | Same noise filter (rejects <4) | ✅ Fixed |
| Surrender → Stand fallback | Strategy returns "R" but app has no Surrender button | `_ACTION_TO_BUTTON["R"]="Surrender"` with cascade to Hit | ✅ Fixed |
| Auto-tap multi-tap (5x/hand) | No de-duplication | Hand tuple + 2.5s cooldown | ✅ Fixed |
| `state=betting` false positive | OCR unreliable on live frame → fell to default `"betting"` | `_colour_buttons_visible()` as primary detector | ✅ Fixed |
| `hands_completed` stuck at 0 | `started_mid_hand` path didn't increment when `hand_counted=False` | Result phase: if `started_mid_hand and not hand_counted` → increment | ✅ Fixed |
| Chip denominations wrong | Detector assigned [5,25,100,500,1000] but app uses [250,500,1K,2.5K,5K] | ⚠️ Not yet fixed — chip OCR reads '250 500 1k 2.5k sk' from strip |
| `_detect_dark_buttons` finds 1 blob | Area threshold too high | ⚠️ Not yet fixed — Deal button IS detected by colour at (870,1994) |

---

## Remaining Issues (next to fix)

### 1. Chip denomination values wrong
`_CHIP_VALUES_DESC = [1000, 500, 100, 25, 5]` — wrong.
App uses `[5000, 2500, 1000, 500, 250]`.
The strip OCR already reads `'250 500 1k 2.5k sk'` — parse these values.
**Fix:** In `_place_bet`, use the OCR'd chip values from `gf.chips` keys directly (the detector populates those from blob colours). Update `_CHIP_VALUES_DESC` to match real app values.

### 2. `_detect_dark_buttons` unreliable (1 blob found, needs 2)
The Deal button at `(870, 1994)` IS detected by the colour method.
**Fix:** Lower Canny area threshold OR trust the colour method and skip `_detect_dark_buttons` for the Deal button.

### 3. `True Count: +0.2` is stuck / not updating visibly
This is likely correct — TC = running_count / decks_remaining. At the start with few cards seen, RC is small and decks_remaining ≈ 6, so TC ≈ 0. Not a bug.

---

## Phone Info

- **Device serial:** `RZCW82D69YH`
- **OS:** Android (USB Debugging enabled)
- **Connection:** USB-C cable
- **Verify:** `adb devices` → `RZCW82D69YH  device`

---

## GitHub Commits

| Commit | Description |
|---|---|
| `0b4509a` | Initial scaffold — all modules, README, 29 tests |
| `e6b67d8` | Game-specific Vegas BJ detector — 5/5 screenshots pass |
| `7045e5d` | Python 3.9 compat fixes, drop pure-python-adb, add setup.py |
| `7898720` | Rich terminal HUD, --verbose on run, auto-resolve brew paths |
| `4d97e08` | fix: betting loop infinite, chip detection robustness |
| `01ec8db` | feat: first-hand manual gate, denomination change with Clear |
| `ab29879` | feat: dark button detection, mid-hand start recovery |
| `9e537bd` | fix: OCR strip checked FIRST in state detection |
| *(uncommitted)* | fix: noise filter, player_display, safe synthetic hand builder |

---

*This file is a backup. If chat is lost, open a new Bob session and reference this file.*
