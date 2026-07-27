# ♠ BJ AI Assistant

> **Real-time BlackJack AI** — reads your Android phone screen via USB, detects your hand and the dealer upcard, computes the mathematically optimal action (Basic Strategy + Hi-Lo counting + Illustrious 18), shows a live terminal HUD, and can **tap the buttons automatically**.

Tested against **Vegas Blackjack** (Android) at 1080×2340 px.

---

## ✦ What it does

| Layer | What happens |
|---|---|
| **Screen Capture** | Grabs frames from your Android phone over USB via ADB (`screencap`) at ~5 FPS |
| **OCR Detection** | Reads the score-bubble totals (dark rounded bubbles above each card pile) — more reliable than card templates |
| **Strategy Engine** | Full Basic Strategy tables (6-deck, Dealer Stands Soft 17, DAS) for Hard / Soft / Pairs |
| **Hi-Lo Counting** | Running count + true count; Illustrious 18 index deviations applied automatically |
| **Terminal HUD** | Live Rich panel in your terminal — no external window needed |
| **Auto-tap** | Sends ADB tap commands directly to the phone — plays hands and places bets automatically |

---

## 🚀 Quick Start

### 1 — Prerequisites

```bash
# macOS (brew)
brew install android-platform-tools tesseract

# Python 3.9+ venv
cd bj-ai-assistant
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2 — Connect your phone

1. Enable **Developer Options** → **USB Debugging** on your Android.
2. Connect via USB cable.
3. Verify: `adb devices` → should show your device serial.

### 3 — Run

```bash
# Live assistant — terminal HUD only (safe, watch-only mode)
bj-assistant run

# Live assistant — verbose debug logs
bj-assistant run -v

# Full automation — program taps the buttons for you
bj-assistant run --auto-tap

# One-shot decision (no phone needed)
bj-assistant decide-cmd --player "A 6" --dealer 5

# Hi-Lo card counting trainer
bj-assistant count

# Diagnose live frame (saves annotated PNG, no inline display)
bj-assistant debug-frame
open /tmp/bj_debug_frame.png
```

---

## 📺 HUD Display

```
╭──────── ♠ BJ AI Assistant ─────────╮
│   🎯 HIT           You: 11  │ Dealer: 6   │
│   True Count: +1.4  Running: +3     Bet: 2× unit  │
│   Hard 11 vs dealer 6 → Hard strategy table        │
╰────────────────────────────────────╯
```

During betting phase:
```
╭──────── ♠ BJ AI Assistant — 🎲 BETTING ─────────╮
│   True Count: +2.1  Running: +5                   │
│   Recommended Bet: 2× unit                        │
│   Placing bet automatically...                    │
╰────────────────────────────────────╯
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
device_serial: null   # null = auto-detect; or "RZCW82D69YH"
fps: 5                # capture frames per second
decks: 6              # decks in shoe
auto_tap: false       # true = tap buttons automatically
show_overlay: true    # show terminal HUD
```

---

## 🃏 Strategy Engine

### Basic Strategy (6-deck S17 DAS)

Full tables for:
- **Hard totals** 5–21
- **Soft totals** A+2 through A+9
- **Pair splits** 2–A

### Hi-Lo Card Counting

| True Count | Bet |
|---|---|
| TC ≤ 1 | 1× unit |
| TC ≤ 2 | 2× unit |
| TC ≤ 3 | 4× unit |
| TC ≤ 4 | 8× unit |
| TC ≥ 5 | 12× unit |

### Illustrious 18 Deviations (subset)

| Hand | Dealer | TC threshold | Action |
|---|---|---|---|
| Hard 16 | 10 | ≥ 0 | Stand |
| Hard 16 | 9 | ≥ 5 | Stand |
| Hard 15 | 10 | ≥ 4 | Stand |
| Hard 12 | 4 | ≥ 0 | Stand |
| Hard 11 | A | ≥ +1 | Double |
| Hard 10 | 10 | ≥ +4 | Double |
| Hard 9 | 2 | ≥ +1 | Double |

### Surrender

| Hand | Dealer | Fallback if no Surrender button |
|---|---|---|
| Hard 16 | 9, 10, A | **Hit** |
| Hard 15 | 10 | **Hit** |

---

## 🤖 Auto-Tap Mode

> No calibration needed — button positions are detected automatically from screen colours.

```bash
bj-assistant run --auto-tap
```

**What happens automatically:**

1. **Betting phase** → detects chip denominations on screen, taps the chip matching the TC-based bet size, then taps Deal
2. **Playing phase** → taps Hit / Stand / Double / Split based on Basic Strategy + count
3. **Result phase** → taps Deal to start the next hand

**Fallback cascade:**
- Surrender not available → Hit
- Double not available → Hit
- Split not available → Hit

**Safety:** 1 tap per unique hand (player+dealer total), 2.5s cooldown between taps.

---

## 🔍 Debug Tool

```bash
bj-assistant debug-frame
open /tmp/bj_debug_frame.png   # annotated regions
open /tmp/bj_raw_frame.png     # plain screenshot
```

Output includes:
- `game_state` (playing / betting / result)
- `dealer_total`, `player_total` from OCR bubbles
- `buttons_found` with pixel counts and pass/fail vs thresholds
- Annotated PNG with all detector region bounding boxes

---

## 📁 Project Structure

```
bj-ai-assistant/
├── bj_assistant/
│   ├── capture.py         # ADB screen capture (auto-finds brew adb)
│   ├── game_detector.py   # Vegas BJ detector — score bubbles + buttons + chips
│   ├── strategy.py        # Basic Strategy tables + HiLoCounter + Illustrious 18
│   ├── overlay.py         # TerminalHUD (Rich) + TkinterHUD (optional)
│   ├── engine.py          # Main loop: capture → detect → decide → display → tap
│   ├── config.py          # YAML settings loader
│   └── cli.py             # CLI: run / decide-cmd / count / calibrate / debug-frame
├── config/settings.yaml
├── tests/
│   ├── test_strategy.py   # 29 unit tests — all pass
│   └── test_capture.py
└── README.md
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap

- [x] Score-bubble OCR (reliable total detection without card templates)
- [x] Button colour detection (state detection without OCR)
- [x] Hi-Lo card counting with Illustrious 18 deviations
- [x] Auto-tap: playing phase (Hit/Stand/Double/Split/Surrender)
- [x] Auto-tap: betting phase (chip selection by TC + Deal button)
- [x] Auto-tap: result phase (tap Deal for next hand)
- [ ] Session stats logger (win/loss, running EV)
- [ ] Live dealer support (video stream OCR)
- [ ] iOS support via QuickTime mirror
- [ ] Side bet advisor (Perfect Pairs, 21+3)

---

## ⚠️ Disclaimer

Educational and research use only. Using software assistance in live casino environments may violate casino terms of service and local regulations. Authors assume no responsibility for misuse.

---

## License

MIT © Adoptsomekids
