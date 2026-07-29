# BJ AI Assistant — Chat Context Backup

> **Date:** 2026-07-22 (updated after live-run analysis + OCR noise fixes)
> **GitHub repo:** https://github.com/Adoptsomekids/bj-ai-assistant
> **Local clone:** `/Users/emilio-ibm/Documents/MOD/BOB/BJ/bj-ai-assistant`
> **Language:** Python 3.9 (venv at `.venv/`)
> **Owner:** Adoptsomekids (GitHub account)

---

## ⚠️ Bob crash prevention rules

1. **NEVER display images inline** — crashes Bob immediately.
2. **Keep responses SHORT** — one `apply_diff` at a time, no code walls.
3. **Read files before editing** — always `read_file` first.
4. **One topic at a time** — confirm each fix before moving to next.

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
6. _bet_placed=True
```

### State machine
```
result   → _bet_placed=False, _bet_phase_entered=0.0
betting tick 1  → init, _bet_placed=False → _place_bet()
_place_bet      → tap chip + 1.2s sleep + tap Deal → _bet_placed=True
betting tick 2+ → _bet_placed=True → skip (no re-tap)
playing  → _bet_phase_entered=0.0 reset
result   → _bet_placed=False again
```

---

## engine.py Key Logic

```
_tick():
  result:  parse win/loss/push → W/L/P stats → auto-tap Deal (green blob)
  betting: if not _bet_placed AND chips visible → _place_bet()
  playing: noise filter → hi-lo count → decide → HUD → tap action

_place_bet():
  tap chip → sleep 1.2s → tap Deal(810,1883) → _bet_placed=True

_execute_action():
  TAP_COOLDOWN=2.5s, de-dup by (player_total, dealer_total)
  Surrender→Hit fallback if Surrender button absent
```

---

## Strategy Decision (engine.py lines 264–315)

```python
# 1. Noise filter: reject player_total < 4 or > 21 (animation OCR glitch)
if player_total is None or not (4 <= player_total <= 21):
    return

# 2. Validate rank OCR: only use if hand_total(ranks) == bubble_total
if gf.player_card_ranks:
    t_check, s_check = hand_total(gf.player_card_ranks)
    _ranks_ok = (t_check == player_total and s_check == is_soft)

# 3. Synthetic hand fallback (when rank OCR fails/partial):
#   player_total==11, no ranks → ["A"]  (soft Ace alone)
#   soft 12-21  → ["A", str(total-11)]  e.g. soft 18 → ["A","7"]
#   hard 12-21  → ["10", str(total-10)] e.g. hard 16 → ["10","6"]
#   hard 4-11   → ["2",  str(total-2)]  e.g. hard  6 → ["2","4"]

# 4. player_display = f"{player_total}{'s' if is_soft else ''}"
#    Always the clean bubble total — never raw synthetic card components
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
- overlay.py line 116: `p_disp = data.get("player_display") or str(data.get("player_total", "?"))`

---

## Verified Working (from 2026-07-22 live run)

From the session log `07:44–07:51`:
- ✅ HIT / STAND decisions correct (6+4=10→HIT, 10+4→STAND vs 6, etc.)
- ✅ True Count shown, Bet 1× unit displayed
- ✅ player_display uses clean total (not synthetic card components)
- ✅ Noise filter rejects player_total=1, 2, 3 (animation glitch)
- ✅ Soft Ace (total=11, no ranks) → ["A"], is_soft=True → strategy uses soft table
- ✅ Hard 11 synthetic hand: ["2","9"] — valid ranks, correct total

### Observed edge cases from live run
| What the HUD showed | What actually happened | Status |
|---|---|---|
| `You: -1 2` | OCR read player_total=1 (glitch) → synthetic `-1,2` | ✅ Fixed: noise filter blocks total<4 |
| `You: As` | player_total=11, no rank OCR | ✅ Fixed: treated as soft Ace ["A"] |
| `You: 2` | OCR read partial total=2 mid-deal | ✅ Fixed: noise filter blocks total<4 |
| `Hard 2 vs dealer 7` | Same: total=2 passed old code | ✅ Fixed |

---

## Fixes History (chronological)

| Bug | Fix |
|---|---|
| `You: -1 2` — OCR noise, total=1 | Noise filter: reject `player_total < 4 or > 21` |
| Synthetic hand invalid ranks (total=1→"-1") | Validated range before building synthetic hand |
| `You: As` display (soft Ace) | `player_total==11, no ranks → ["A"], is_soft=True` |
| Partial rank OCR (A+9→total=11) | Validate `hand_total(ranks)==bubble_total` before using ranks |
| HUD shows raw card components (`4 2`) | `player_display = f"{total}{'s' if soft else ''}"` |
| Dealer Ace shows as `1` | `dealer_total==1 → "A"` in `effective_dealer_upcard` |
| No phone connected → macOS screencap | `get_best_capture` fails fast with clear ADB error |
| `hands_completed` stuck at 0 | Removed first-hand gate; `_result_handled` flag per result |
| Chip denominations wrong (5/25/100) | `CHIP_VALUES=[250,500,1000,2500,5000]` |
| Chip zone included Clear/Deal blobs | `CHIP_ROW_Y_TOP=0.880` (was 0.750) |
| Double chip tap (250+500=750) | `_place_bet` atomic: chip + 1.2s sleep + Deal; no re-entry |
| `bet_is_placed` always True/False | Replaced with atomic `_place_bet` — no OCR polling needed |
| Balance OCR reads wrong value | `max(candidates)` from x=0.10-0.42 region |
| Balance unknown → unbound bet | Safe cap: `desired=250` when `balance=None` |
| Game not in foreground → blind tap | Guard: skip bet if `strip_text=''` AND no chips AND no bet |
| Infinite re-trigger in betting phase | Do NOT reset `_bet_phase_entered` inside `_place_bet` |

---

## Known Remaining Issues

1. **Balance OCR unreliable**: reads `1,501` instead of `4,501` (coin icon at x<0.17). `max(candidates)` helps but may still be off. Not critical — bet cap uses conservative values.
2. **Result phase Deal**: uses `_detect_deal_button` (green blob). Works for post-result Deal but may need fixed-position fallback.
3. **W/L/P stats**: OCR of result banner reads `'dealer wins'`/`'you win'` etc. May miss some results at 5fps.
4. **True Count display**: always shows `+0.2` — counter is updating (RC logged) but TC display may be off if `decks` config is wrong. Should show integer-like values mid-shoe.

---

## Phone Info

- **Serial:** `RZCW82D69YH`
- **Verify:** `adb devices` → `RZCW82D69YH  device`

---

## Recent Git Commits

| Commit | Description |
|---|---|
| latest | fix: OCR noise filter, synthetic hand safety, player_display clean total |
| `fe49d32` | fix: atomic bet — chip + 1.2s wait + Deal at fixed position |
| `c38e772` | fix: two-phase bet attempt (superseded by fe49d32) |
| `01cf250` | fix: bet_is_placed uses action strip OCR |
| `66ff95d` | fix: skip bet when game not in foreground |
| `3f6977e` | fix: dealer Ace bubble reads as 1; partial rank OCR |
| `d95ffaf` | fix: auto-play immediate; W/L/P session stats |
| `c36a6da` | fix: fail fast when phone not connected |

---

## Future: AI Agent Upgrade

User wants to evolve toward a fully autonomous AI casino agent (RL/ML).
Relevant repos to clone to `/Users/emilio-ibm/Documents/MOD/BOB/BJ`:
- https://github.com/tarunravi/BlackjackAI (RL agent)
- https://github.com/GregSommerville/machine-learning-blackjack-solution (ML)

---

*Backup. Open new Bob session, paste this file, continue.*
