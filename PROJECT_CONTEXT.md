# BJ AI Assistant — Project Context Backup

> **Created:** 2026-07-22  
> **GitHub repo:** https://github.com/Adoptsomekids/bj-ai-assistant  
> **Local clone:** `/Users/emilio-ibm/Documents/MOD/BOB/BJ/bj-ai-assistant`  
> **Language:** Python 3.10+  
> **Owner:** Adoptsomekids (GitHub)

---

## What We Are Building

An intelligent real-time BlackJack AI assistant system that:

1. **Mirrors the Android phone screen** to the Mac via USB cable using ADB (`adb exec-out screencap -p`).
2. **Detects playing cards** in the captured frames using OpenCV template matching + Tesseract OCR fallback.
3. **Computes the optimal action** using a complete Basic Strategy table (6-deck, S17, DAS) plus Hi-Lo card counting with Illustrious 18 deviations.
4. **Displays a live HUD overlay** on the Mac — a borderless always-on-top Tkinter window showing: action (HIT/STAND/DOUBLE/SPLIT/SURRENDER), true count, bet sizing recommendation, and reasoning.
5. **(Optional / experimental)** Sends ADB tap commands directly to the phone to automate button presses after calibrating button pixel positions.

---

## Architecture Overview

```
Phone (Android) ──USB──▶ ADB ──▶ capture.py
                                      │
                                      ▼
                              card_detector.py
                                      │
                                      ▼
                               strategy.py
                               HiLoCounter
                                      │
                             ┌────────┴────────┐
                             ▼                 ▼
                         overlay.py       adb.tap()
                        (HUD on Mac)    (auto-click)
                             │
                         engine.py  ◀── cli.py
```

---

## Module Breakdown

| File | Purpose |
|---|---|
| `capture.py` | Screen capture: `ADBCapture` (USB/TCP), `ScrcpyCapture` (OpenCV VideoCapture), `MacOSWindowCapture` (fallback). Factory: `get_best_capture()` |
| `card_detector.py` | `TemplateCardDetector` (PNG atlas matching), `OCRCardDetector` (Tesseract), `CardDetector` (combined). NMS dedup. Role assignment (dealer=top 40%, player=bottom 60%) |
| `strategy.py` | Full Basic Strategy tables (hard/soft/pairs). `HiLoCounter` (running+true count). `decide()` function + Illustrious 18 count deviations. Bet sizing 1×–12× |
| `overlay.py` | `HUDOverlay` — borderless Tkinter window, draggable, always-on-top, thread-safe `update()` |
| `engine.py` | `BJEngine` — main loop thread, ties all modules together, exposes `start()/stop()` |
| `config.py` | YAML loader → `Settings` dataclass |
| `cli.py` | Click CLI: `run`, `decide`, `count`, `calibrate` |

---

## Key Technical Decisions

- **Python** chosen over Go for this project because OpenCV, Tesseract, and Tkinter ecosystem is Python-native.
- **ADB over USB** is the primary capture method — most reliable, no extra software, works with any Android game.
- **scrcpy** is the recommended companion for best frame rate (~30fps) — run separately, then point the engine at the OpenCV VideoCapture feed.
- **Template matching** is the primary card detection method — requires a one-time setup of a card template atlas (PNG images for each of 52 cards in the game's visual skin). OCR is the fallback.
- **No ML model required** for v1 — template matching + OCR is sufficient. YOLOv8 fine-tuning is on the roadmap for v2.
- **Auto-tap is off by default** — requires manual calibration of button pixel coordinates per game/resolution.

---

## Setup Steps (recap)

```bash
# 1. Prerequisites
brew install android-platform-tools tesseract scrcpy

# 2. Python env
cd /Users/emilio-ibm/Documents/MOD/BOB/BJ/bj-ai-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Connect phone: USB Debugging on → adb devices

# 4. Run
bj-assistant run

# One-shot (no phone needed)
bj-assistant decide --player "A 6" --dealer 5
```

---

## Card Templates — Next Step

The system needs a card template atlas to do visual detection. Two options:

1. **Manual**: Screenshot each card in the target game app, crop to just the card, save as `AH.png`, `KS.png`, `10D.png` etc. in `assets/card_templates/`.
2. **Auto-generator** (roadmap): Script that cycles through a test BJ hand and screenshots each card automatically.

Until templates exist the system falls back to OCR (Tesseract), which is slower but functional.

---

## Environment Notes

- Mac: Apple Silicon (arm64), macOS Sequoia
- Python: 3.10+
- ADB: via `android-platform-tools` (Homebrew)
- Phone: Android with USB Debugging enabled

---

## Roadmap (as discussed)

- [ ] Card template atlas generator
- [ ] iOS support (`pymobiledevice3` + QuickTime)
- [ ] Live dealer casino video stream analysis
- [ ] Side bet advisor (Perfect Pairs, 21+3)
- [ ] Session stats + EV tracker
- [ ] Web dashboard (FastAPI + htmx)
- [ ] YOLOv8 fine-tune for unknown card skins

---

*This file is a backup context document. If this chat is lost, start a new session and reference this file to resume work.*
