# BJ AI Assistant

> **Real-time BlackJack AI assistant** — card detection, optimal Basic Strategy, Hi-Lo card counting, and live HUD overlay for mobile BlackJack games via USB/ADB mirroring.

---

## ✦ What it does

| Layer | What happens |
|---|---|
| **Screen Capture** | Grabs frames from your Android phone over USB via ADB (or scrcpy/macOS window fallback) |
| **Card Detection** | Identifies all visible cards using template matching + OCR fallback |
| **Strategy Engine** | Computes the mathematically optimal action using full Basic Strategy tables (6-deck S17 DAS) |
| **Hi-Lo Counting** | Tracks the running/true count and applies the Illustrious 18 deviations |
| **HUD Overlay** | Renders a borderless, always-on-top HUD on your Mac with the decision, bet sizing, and count |
| **Auto-tap** *(opt.)* | Optionally sends the tap via ADB directly to the phone button |

---

## 📁 Project Structure

```
bj-ai-assistant/
├── bj_assistant/
│   ├── __init__.py        # Package metadata
│   ├── capture.py         # Screen capture backends (ADB, scrcpy, macOS)
│   ├── card_detector.py   # Card detection (template matching + OCR)
│   ├── strategy.py        # Basic Strategy tables + Hi-Lo counter + deviations
│   ├── overlay.py         # Tkinter HUD overlay (always-on-top, draggable)
│   ├── engine.py          # Main orchestration loop
│   ├── config.py          # YAML config loader
│   └── cli.py             # Click CLI (run / decide / count / calibrate)
├── assets/
│   └── card_templates/    # PNG card templates (52 cards × skins)
├── config/
│   └── settings.yaml      # Runtime configuration
├── tests/
│   ├── test_strategy.py   # Strategy + counting unit tests
│   └── test_capture.py    # Capture backend unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 — Prerequisites

```bash
# macOS
brew install android-platform-tools tesseract

# Python deps
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### 2 — Connect your phone

1. Enable **Developer Options** on your Android phone.
2. Enable **USB Debugging**.
3. Connect via USB cable.
4. Verify: `adb devices` should show your device.

### 3 — Run the assistant

```bash
# Start real-time mode (camera + HUD overlay)
bj-assistant run

# One-shot decision (no camera needed)
bj-assistant decide --player "A 6" --dealer 5

# Hi-Lo card counting trainer
bj-assistant count

# Calibrate button positions for auto-tap
bj-assistant calibrate
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
device_serial: null        # null = auto-detect
fps: 5                     # capture frames per second
decks: 6                   # decks in shoe
auto_tap: false            # enable ADB auto-tap (requires calibration)
show_overlay: true         # show HUD on Mac
```

---

## 🃏 Strategy Engine

### Basic Strategy

Full 6-deck, Dealer Stands on Soft 17, Double After Split (DAS) tables for:
- **Hard totals** (5–21)
- **Soft totals** (A+2 through A+9)
- **Pair splits** (2–2 through A–A)

### Hi-Lo Card Counting

- Tracks running count and true count (running ÷ decks remaining)
- Bet sizing: 1× unit at TC ≤ 1, up to 12× at TC ≥ +5
- **Illustrious 18 deviations** applied automatically (e.g. Stand 16 vs 10 at TC ≥ 0, Double 11 vs A at TC ≥ +1, etc.)

### Actions

| Code | Meaning |
|---|---|
| `H` | Hit |
| `S` | Stand |
| `D` | Double Down |
| `P` | Split |
| `R` | Surrender (Hit if unavailable) |

---

## 📱 Phone Mirror Setup (USB — recommended)

```bash
# Option A: scrcpy (best performance, ~30fps)
brew install scrcpy
scrcpy --window-title BJMirror &
bj-assistant run

# Option B: pure ADB (no additional software, ~2-5fps)
bj-assistant run --fps 3
```

---

## 🤖 Auto-Tap Mode

> ⚠️ **Experimental.** Requires calibrating button pixel coordinates first.

```bash
# Step 1 — open BJ game on phone, then:
bj-assistant calibrate

# Step 2 — update config/settings.yaml with the printed coordinates

# Step 3 — run with auto-tap enabled
bj-assistant run --auto-tap
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🗺️ Roadmap

- [ ] Card template atlas generator (auto-screenshots all 52 cards per skin)
- [ ] iOS support via `pymobiledevice3` + `QuickTime` mirror
- [ ] Live dealer casino support (video stream analysis)
- [ ] Side bet advisor (Perfect Pairs, 21+3)
- [ ] Session stats logger (win/loss tracking, EV calculator)
- [ ] Web dashboard (FastAPI + htmx)
- [ ] ML-based card detector (YOLOv8 fine-tune) for unknown card skins

---

## ⚠️ Disclaimer

This tool is intended **for educational and research purposes only**. Using software assistance in live casino games may violate casino rules and local gambling regulations. The authors assume no responsibility for any misuse.

---

## License

MIT © Adoptsomekids
