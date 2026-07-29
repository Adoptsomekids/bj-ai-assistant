# BJ AI Assistant — Chat Context Backup

> **Date:** 2026-07-22 (updated — AI layer implemented)
> **GitHub repo:** https://github.com/Adoptsomekids/bj-ai-assistant
> **Local clone:** `/Users/emilio-ibm/Documents/MOD/BOB/BJ/bj-ai-assistant`
> **Language:** Python 3.9 (venv at `.venv/`)
> **Commit:** `62ac88b` — feat: AI layer — Monte Carlo Q-table + Kelly Criterion bet sizing

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
5. **AI layer**: Monte Carlo Q-table + Kelly Criterion bet sizing
6. Show live terminal HUD (Rich) with AI/BS badge
7. **Auto-tap**: betting → chip + Deal, playing → Hit/Stand/Double/Split, result → Deal

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

bj-assistant run --auto-tap     # full automation with AI
bj-assistant run                # advise only
bj-assistant train-ai --episodes 500000   # retrain Q-table (better model)
bj-assistant -v debug-frame     # capture frame, print detection report
```

---

## Repo Structure

```
bj-ai-assistant/
├── bj_assistant/
│   ├── engine.py          ★ main loop — integrates AI + Kelly
│   ├── game_detector.py   ★ VegasBJDetector, GameFrame, Layout constants
│   ├── overlay.py         — TerminalHUD (Rich), shows [AI] or [BS] badge
│   ├── strategy.py        — Basic Strategy tables, HiLoCounter, Illustrious 18
│   ├── cli.py             — CLI: run / decide-cmd / count / train-ai / debug-frame
│   ├── capture.py         — ADBCapture (fails fast with clear error if no device)
│   ├── config.py
│   └── ai/
│       ├── __init__.py
│       ├── mc_trainer.py  ★ Monte Carlo RL trainer (offline)
│       ├── kelly.py       ★ Kelly Criterion + Hi-Lo bet sizing
│       └── ai_advisor.py  ★ Runtime Q-table decision layer
├── models/
│   └── bj_qtable.npy      — Pre-trained 100k episode Q-table (225 KB)
├── config/settings.yaml
├── tests/test_strategy.py  (29 passing)
└── CHAT_CONTEXT.md
```

---

## AI Architecture

### Decision cascade (runtime)
```
1. Basic Strategy + Illustrious 18 deviations (always runs)
2. AI Advisor (Q-table lookup):
   - Agrees with BS → show [AI✓] confirmation in HUD
   - Disagrees with BS → log "AI≠BS, BS wins", keep BS decision
   (BS wins on disagreements: proven mathematically optimal)
3. Kelly Criterion → bet sizing (replaces simple TC spread)
```

### Monte Carlo Q-table (`mc_trainer.py`)
```
State: (player_total, is_soft, dealer_upcard, tc_bucket, can_double, can_split)
  = 18 × 2 × 10 × 5 × 2 × 2 = 7,200 discrete states
Actions: S / H / D / P  (4 actions)
Algorithm: Every-Visit Monte Carlo, epsilon-greedy (ε=0.10, α=0.02, γ=1.00)
Source: Adapted from tarunravi/BlackjackAI Monte Carlo approach

Training: bj-assistant train-ai --episodes 500000  (~2 min on M-series)
Current: models/bj_qtable.npy (100k episodes, 225 KB, ~50% BS agreement)
Target:  500k+ episodes for >90% BS agreement
```

### Kelly Criterion (`kelly.py`)
```
Player edge formula (Stanford Wong, "Professional Blackjack"):
  edge(TC) = (TC - 1) × 0.5%   [TC pivot = +1 for Hi-Lo 6-deck S17]

Kelly fraction: f* = edge / BJ_variance (1.15)
Fractional Kelly: /4 (conservative, reduces risk of ruin)

Practical bet spread (adapted for chip sizes):
  TC ≤ +2  →  1 unit  ($250)   — house has edge
  TC = +3  →  2 units ($500)   — Kelly 2×
  TC = +4  →  4 units ($1000)  — Kelly 4×
  TC = +5  →  6 units ($1000)  — nearest chip
  TC ≥ +6  → 10 units ($2500)  — max cap
```

### Why AI disagrees early (important)
- At 100k episodes, Q-table has ~50% agreement with BS
- Needs 500k-1M episodes for >90% agreement (MC convergence)
- **Design decision: BS always wins until AI proves itself**
- AI disagreements are logged for analysis → will add override at 90%+ confidence

---

## Layout Constants (game_detector.py — verified on 1080×2340)

| Region | Fraction | Absolute px | Notes |
|---|---|---|---|
| Dealer bubble | cx=0.500, cy=0.186 | (540,436) | score bubble OCR |
| Player bubble | cx=0.500, cy=0.708 | (540,1657) | score bubble OCR |
| Button row | y=0.877–0.962 | 2052–2251 | Stand/Hit/Double/Split |
| Chip row | y=0.880–0.975 | 2059–2281 | 250/500/1K/2.5K/5K chips |
| Balance | x=0.10–0.42, y=0.025–0.115 | | `max()` of OCR candidates |
| **Deal button** | **x=0.75w, y=0.805h** | **(810,1883)** | **fixed position — verified** |

---

## engine.py Key Logic

```
_tick():
  result:   parse win/loss/push → stats → auto-tap Deal (green blob)
  betting:  if not _bet_placed AND chips visible → _place_bet()
            Kelly Criterion → kelly_chip_amount(tc, balance)
  playing:  noise filter (4≤total≤21) → hi-lo count → decide (BS+I18)
            → AI advisor confirms/disagrees → tap action

_place_bet():
  Kelly chip amount → tap chip → sleep 1.2s → tap Deal(810,1883) → _bet_placed=True

_execute_action():
  TAP_COOLDOWN=2.5s, de-dup by (player_total, dealer_total)
  Surrender→Hit fallback if Surrender button absent
```

---

## Strategy Decision Logic (engine.py lines ~264–370)

```python
# 1. Noise filter: reject player_total < 4 or > 21
if player_total is None or not (4 <= player_total <= 21): return

# 2. Validate rank OCR: only use if hand_total(ranks) == bubble_total
_ranks_ok = (t_check == player_total and s_check == is_soft)

# 3. Synthetic hand fallback:
#   player_total==11, no ranks → ["A"]  (soft Ace alone)
#   soft 12-21  → ["A", str(total-11)]
#   hard 12-21  → ["10", str(total-10)]
#   hard 4-11   → ["2",  str(total-2)]

# 4. Basic Strategy + I18 → bs_action

# 5. AI Advisor → ai_action
#   if ai_action == bs_action: add [AI✓] to reasoning
#   if ai_action != bs_action: log disagreement, keep bs_action

# 6. player_display = f"{player_total}{'s' if is_soft else ''}"
```

---

## HUD Display

```
╭──── ♠ BJ AI Assistant ────────────╮
│  🎯 HIT    You: 16   │ Dealer: 10  │
│  TC: +2.1  RC: +12       Bet: 1×  [AI]
│  W:5 L:3 P:1 #9    Hard 16 vs 10 [AI✓]
╰────────────────────────────────────╯
```
- `[AI]` green badge = Q-table model loaded
- `[BS]` grey badge = no model / fallback
- `[AI✓]` in reasoning = AI agrees with BS
- `[AI:H≠BS, BS wins]` = disagreement logged

```
╭──── ♠ BJ AI Assistant — 🎲 BETTING ╮
│  TC: +3.0  RC: +18                  │
│  Bet: Kelly 2× unit  ($500)         │
│  W:5 L:3 P:1  Hands:9               │
│  🤖 AI+Kelly ON                     │
╰─────────────────────────────────────╯
```

---

## Repo Analysis Results

| Repo | What it is | Useful? |
|---|---|---|
| **tarunravi/BlackjackAI** | Monte Carlo RL, Python + NumPy, Pygame UI | ✅ **Algorithm adapted** for mc_trainer.py |
| **GregSommerville/ml-bj-solution** | Genetic Algorithm in C#, evolves strategy tables | ❌ C#, GA not practical to port |
| **Whale-io/lets-play-a-game** | MCP server for AI agents on Whale.io crypto casino, $10k prize pool | 🔥 **FUTURE: Enter Season 1 tournament** |
| **MoonshotAI/Kimi-K3** | 2.8T parameter LLM, open weights | ❌ Too large for local real-time decisions |
| **ArtBreguez/Double-AI-Bot** | Blaze casino bet signal predictor (not BJ) | ❌ Different game |
| **egorfedorov/Slot-Casino** | Stake Engine slot development AI skills | ❌ Slot-specific |

### Whale.io Opportunity 🐋
- Season 1: $10,000 prize pool, 14-day AI agent tournament
- Supports Blackjack via MCP: `whale_blackjack` game code
- Connect: `claude mcp add whale-games --transport http https://api.playwhale.io/mcp`
- **Our agent could enter**: integrate OpenClaw MCP + adapt BJ engine to Whale.io API

---

## Fixes History (complete)

| Bug | Fix |
|---|---|
| `You: -1 2` — OCR noise | Noise filter: reject `player_total < 4 or > 21` |
| Synthetic hand invalid ranks | Safe synthetic: hard 4-11 → ["2", str(total-2)] |
| `You: As` display (soft Ace) | `player_total==11, no ranks → ["A"], is_soft=True` |
| HUD shows raw card components | `player_display = f"{total}{'s' if soft else ''}"` |
| Dealer Ace shows as `1` | `dealer_total==1 → "A"` in `effective_dealer_upcard` |
| No phone → macOS screencap | `get_best_capture` fails fast with ADB error |
| `hands_completed` stuck at 0 | Removed first-hand gate; `_result_handled` flag |
| Chip denominations wrong | `CHIP_VALUES=[250,500,1000,2500,5000]` |
| Double chip tap | `_place_bet` atomic: chip + 1.2s + Deal; no re-entry |
| Infinite re-trigger | Do NOT reset `_bet_phase_entered` inside `_place_bet` |
| Simple TC spread replaced | Kelly Criterion: `f* = edge / 1.15 / 4` |

---

## Known Remaining Issues

1. **Balance OCR unreliable**: reads `1,501` vs `4,501` (coin icon). `max(candidates)` helps.
2. **Result phase Deal**: green blob detection. Works but may need fixed position fallback.
3. **W/L/P stats**: OCR result banner may miss at 5fps.
4. **True Count always +0.2**: correct at start of shoe (few cards counted). Increases mid-shoe.
5. **Q-table ~50% agreement**: needs 500k+ episodes for full convergence. Train: `bj-assistant train-ai --episodes 500000`

---

## Recent Git Commits

| Commit | Description |
|---|---|
| `62ac88b` | feat: AI layer — Monte Carlo Q-table + Kelly Criterion bet sizing |
| (prev) | fix: OCR noise filter, synthetic hand safety, player_display clean total |
| `fe49d32` | fix: atomic bet — chip + 1.2s wait + Deal at fixed position |
| `3f6977e` | fix: dealer Ace bubble reads as 1; partial rank OCR |
| `d95ffaf` | fix: auto-play immediate; W/L/P session stats |
| `c36a6da` | fix: fail fast when phone not connected |

---

## Next Steps

### Immediate
1. **Train Q-table more**: `bj-assistant train-ai --episodes 500000` → ~2 min, >80% BS agreement
2. **Run live test**: use `bj-assistant run --auto-tap` and watch `[AI✓]` vs `[AI:X≠BS]` ratio

### Near Future
3. **Whale.io agent**: adapt engine for MCP `whale_blackjack` API + enter Season 1 ($10k prize)
4. **Composition-dependent strategy**: precompute EV table for hard 12-16 vs dealer 2-10 (~2h work, +0.3% EV)
5. **Improve OCR**: player card rank detection for better composition-aware decisions

---

*Backup. Open new Bob session, paste this file, continue.*
