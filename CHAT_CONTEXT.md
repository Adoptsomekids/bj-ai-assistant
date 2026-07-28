# BJ AI Assistant — Chat Context Backup

> **Date:** 2026-07-28 (updated after extensive betting phase debugging)
> **GitHub repo:** https://github.com/Adoptsomekids/bj-ai-assistant
> **Local clone:** `/Users/emilio-ibm/Documents/MOD/BOB/BJ/bj-ai-assistant`
> **Language:** Python 3.9 (venv at `.venv/`)
> **Owner:** Adoptsomekids (GitHub account)

---

## ⚠️ Bob crash prevention rules

1. **NEVER display images inline** — crashes Bob immediately.
2. **Keep responses SHORT** — one `apply_diff` at a time.
3. **Read files before editing** — always `read_file` first.

---

## Project Goal

Real-time BlackJack AI assistant:
1. ADB screencap from Android phone over USB
2. Detect game state (betting / playing / result)
3. OCR hand totals from score bubbles
4. Compute optimal action: Basic Strategy + Hi-Lo counting + Illustrious 18
5. Show live terminal HUD (Rich)
6. **Auto-tap**: betting → chip + Deal, playing → Hit/Stand/Double/Split, result → Deal

### Automation pipeline
```
betting:  tap chip (TC-scaled) → wait 1.2s → tap Deal at (0.75w, 0.805h)
playing:  tap optimal action (Basic Strategy + Hi-Lo + I18)
result:   tap Deal at green button (bottom of screen)
→ loops infinitely until Ctrl+C
```

---

## Target App / Device

- **App:** Vegas Blackjack (Android), Dealer Stands Soft 17, BJ pays 3:2
- **Resolution:** 1080 × 2340 px (portrait)
- **Device:** serial `RZCW82D69YH`, USB debugging enabled
- **Chip values:** 250, 500, 1000, 2500, 5000

---

## Setup & Run

```bash
adb devices   # → RZCW82D69YH  device
cd ~/Documents/MOD/BOB/BJ/bj-ai-assistant
source .venv/bin/activate
bj-assistant run --auto-tap     # full automation
bj-assistant run                # advise only
bj-assistant -v debug-frame     # capture frame, print detection report
```

---

## Repo Structure

```
bj-ai-assistant/
├── bj_assistant/
│   ├── engine.py          ★ main loop, betting/playing/result phases
│   ├── game_detector.py   ★ VegasBJDetector, GameFrame, Layout constants
│   ├── overlay.py         — TerminalHUD (Rich), W/L/P stats display
│   ├── strategy.py        — Basic Strategy tables, HiLoCounter, Illustrious 18
│   ├── cli.py             — Click CLI: run / decide-cmd / debug-frame
│   ├── capture.py         — ADBCapture (fails fast with clear error if no device)
│   └── config.py
├── config/settings.yaml
├── tests/test_strategy.py  (29 passing)
└── CHAT_CONTEXT.md
```

---

## Layout Constants (game_detector.py — verified on 1080×2340)

| Region | Fraction | Absolute px | Notes |
|---|---|---|---|
| Dealer bubble | cx=0.500, cy=0.186 | (540,436) | score bubble OCR |
| Player bubble | cx=0.500, cy=0.708 | (540,1657) | score bubble OCR |
| Button row | y=0.877–0.962 | 2052–2251 | Stand/Hit/Double/Split |
| Chip row | y=0.880–0.975 | 2059–2281 | 250/500/1K/2.5K/5K chips |
| Balance | x=0.10–0.42, y=0.025–0.115 | | `max()` of OCR candidates |
| Clear button | x=0.25w, y=0.805h | (270,1883) | fixed position |
| **Deal button** | **x=0.75w, y=0.805h** | **(810,1883)** | **fixed position — verified** |
| Bet amount zone | y=0.688–0.718 | 1609–1680 | shows "250" when chip on table |

### Chip positions (detected by gold blob in y=0.880–0.975)
| Chip | Typical x | Typical y |
|---|---|---|
| 250 | 65 | 2224 |
| 500 | 161 | 2126 |
| 1000 | ~729 | ~2116 |
| 2500 | ~918 | ~2126 |

---

## Button Color Detection (HSV)

| Button | HSV Low | HSV High | Min px |
|---|---|---|---|
| Stand | (0,100,80) | (12,255,255) | 5000 |
| Hit | (50,60,80) | (90,255,255) | 400 |
| Double | (95,80,80) | (135,255,255) | 5000 |
| Split | (12,100,80) | (28,255,255) | 15000 |

Playing state primary detector: bright-green Hit px ≥ 3000.

---

## Betting Phase — Final Working Design

### `_place_bet()` — atomic sequence
```python
1. Compute desired = _target_bet_amount(gf.balance)
   TC≤1→250, TC≤2→500, TC≤3→500, TC≤4→1000, TC≥5→2500
   Capped to balance (safe_default=250 if balance unknown)
2. Pick closest chip ≤ desired from gf.chips
3. Tap chip at detected position
4. Sleep 1.2s  (app animation time)
5. Tap Deal at FIXED position (0.75w, 0.805h) = (810, 1883)
6. _bet_placed=True, _bet_phase_entered=0.0
```

### Why fixed Deal position
- `_detect_deal_button` (color-based) finds green blob at (350,2161) which is NOT the Deal button — it's part of the chip area.
- The real Deal button (dark background, white text "DEAL") is always at (0.75w, 0.805h) after a chip is tapped.
- Verified from live OCR: `y=0.80-0.85 right half → 'deal'` at x≈0.75.

### bet_is_placed detection (not used for tapping anymore)
- Was: OCR of various zones — all unreliable
- Strip `y=0.877` always shows `'250 500 1k...'` regardless
- Zone `y=0.80` always shows `'clear deal'` regardless
- Bet amount at `y=0.688` works but OCR unreliable
- **Current approach: atomic tap in `_place_bet`, no polling needed**

---

## engine.py Key Logic

```
_tick():
  result:  parse win/loss/push → W/L/P stats → auto-tap Deal (green)
  betting: if not _bet_placed AND chips visible → _place_bet()
  playing: noise filter → hi-lo count → decide → HUD → tap action

_place_bet():
  tap chip → sleep 1.2s → tap Deal(810,1883) → _bet_placed=True

_execute_action():
  TAP_COOLDOWN=2.5s, de-dup by (player_total, dealer_total)
  Surrender→Hit fallback
```

---

## HUD Display

```
╭──── ♠ BJ AI Assistant ─────────────╮
│  🎯 HIT    You: 11   │ Dealer: 6   │
│  TC: +1.2  RC: +6        Bet: 2×   │
│  W:3 L:2 P:1 #6    Hard 11 vs 6   │
╰────────────────────────────────────╯
```
- `You: 16` = hard 16, `You: 18s` = soft 18, `You: 11s` = Ace alone

---

## Fixes History (chronological)

| Bug | Fix |
|---|---|
| `You: -1 2` — OCR noise | Noise filter: reject player_total < 4 |
| Dealer Ace shows as `1` | `dealer_total==1 → "A"` in `effective_dealer_upcard` |
| Partial rank OCR (A+9 → total=11) | Validate `hand_total(ranks)==bubble_total` before using ranks |
| No phone connected → macOS screencap | `get_best_capture` fails fast with clear ADB error message |
| `hands_completed` stuck at 0 | Removed first-hand gate; `_result_handled` flag per result screen |
| Chip denominations wrong (5/25/100) | `CHIP_VALUES=[250,500,1000,2500,5000]` |
| Chip zone included Clear/Deal blobs | `CHIP_ROW_Y_TOP=0.880` (was 0.750) excludes y≈1821 buttons |
| Clear/Deal Canny detection unreliable | Use fixed positions (0.25w,0.805h) and (0.75w,0.805h) |
| Deal tap opened store (wrong coords) | Deal at fixed (0.75w,0.805h), NOT color-detected blob |
| Double chip tap (250+500=750) | `_place_bet` atomic: chip + 1.2s sleep + Deal; no re-entry |
| `bet_is_placed` always True/False | Replaced with atomic `_place_bet` — no OCR polling needed |
| Balance OCR reads `41,507` vs `4,501` | `max(candidates)` from x=0.10-0.42 region |
| Balance unknown → unbound bet | Safe cap: `desired=250` when `balance=None` |
| Game not in foreground → blind tap | Guard: skip bet if `strip_text=''` AND no chips AND no bet |

---

## Known Remaining Issues

1. **Balance OCR unreliable**: reads `1,501` instead of `4,501` due to coin icon at x<0.17. `max(candidates)` helps but may still be off. Not critical — bet cap uses conservative values.
2. **result phase Deal detection**: uses `_detect_deal_button` (green blob) which finds (350,2161). This works for the post-result Deal but may need the same fixed-position approach if it fails.
3. **Win/Loss/Push stats**: OCR of result banner reads `'dealer wins'`/`'you win'` etc. May miss some results at 5fps.

---

## Phone Info

- **Serial:** `RZCW82D69YH`
- **Verify:** `adb devices` → `RZCW82D69YH  device`

---

## Recent Git Commits

| Commit | Description |
|---|---|
| `fe49d32` | fix: atomic bet — chip + 1.2s wait + Deal at fixed position |
| `c38e772` | fix: two-phase bet attempt (superseded by fe49d32) |
| `01cf250` | fix: bet_is_placed uses action strip OCR |
| `66ff95d` | fix: skip bet when game not in foreground |
| `fc38101` | fix: Clear/Deal by OCR at y=0.77-0.84 |
| `783dd23` | fix: balance OCR x=0.17-0.32; safe cap when unknown |
| `b5caa9f` | fix: chip zone y=0.88 excludes Clear/Deal buttons |
| `3f6977e` | fix: dealer Ace bubble reads as 1; partial rank OCR |
| `d95ffaf` | fix: auto-play immediate; W/L/P session stats |
| `c36a6da` | fix: fail fast when phone not connected |

---

*Backup. Open new Bob session, reference this file, continue.*
