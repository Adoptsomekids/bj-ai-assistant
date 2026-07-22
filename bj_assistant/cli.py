"""
cli.py
------
Command-line interface for BJ AI Assistant.

Commands:
  run       Start the real-time assistant (capture + detect + advise)
  decide    One-shot decision for manually entered cards
  count     Interactive card counting trainer
  calibrate Help calibrate button positions for auto-tap
"""

from __future__ import annotations
import logging
import sys
import time

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .config import load_settings
from .strategy import GameState, HiLoCounter, decide, hand_total
from .engine import BJEngine
from .capture import ADBCapture

console = Console()

ACTION_STYLES = {
    "H": "bold red",
    "S": "bold green",
    "D": "bold yellow",
    "P": "bold blue",
    "R": "bold magenta",
}


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", default=None, help="Path to settings.yaml")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """♠ BJ AI Assistant — real-time BlackJack strategy engine."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    from pathlib import Path
    cfg_path = Path(config) if config else None
    ctx.ensure_object(dict)
    ctx.obj["settings"] = load_settings(cfg_path) if cfg_path else load_settings()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--serial", default=None, help="ADB device serial (optional)")
@click.option("--fps", default=None, type=int, help="Capture FPS (default: from config)")
@click.option("--auto-tap", is_flag=True, help="Enable ADB auto-tap (requires calibration)")
@click.option("--no-overlay", is_flag=True, help="Disable terminal HUD overlay")
@click.option("--float-hud", is_flag=True, help="Use floating Tkinter window (if available)")
@click.option("--decks", default=None, type=int, help="Number of decks in shoe")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def run(ctx: click.Context, serial: str, fps: int, auto_tap: bool,
        no_overlay: bool, float_hud: bool, decks: int, verbose: bool) -> None:
    """Start the real-time BJ assistant."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    s = ctx.obj["settings"]
    engine = BJEngine(
        device_serial=serial or s.device_serial,
        fps=fps or s.fps,
        auto_tap=auto_tap or s.auto_tap,
        show_overlay=not no_overlay and s.show_overlay,
        use_tkinter=float_hud,
        decks=decks or s.decks,
    )
    console.print(Panel(
        "[bold green]BJ AI Assistant starting...[/bold green]\n"
        f"FPS: {fps or s.fps}  |  Auto-tap: {auto_tap}  |  Decks: {decks or s.decks}\n"
        "Press [bold]Ctrl+C[/bold] to stop.",
        title="♠ BJ AI Assistant", border_style="green"
    ))
    try:
        engine.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        engine.stop()


# ---------------------------------------------------------------------------
# decide (manual one-shot)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--player", "-p", required=True, help='Player cards, e.g. "A 6"')
@click.option("--dealer", "-d", required=True, help='Dealer upcard, e.g. "7"')
@click.option("--count", "-c", default=0, type=float, help="Current true count (optional)")
@click.option("--decks", default=6, type=int, help="Number of decks")
@click.pass_context
def decide_cmd(ctx: click.Context, player: str, dealer: str, count: float, decks: int) -> None:
    """Get the optimal decision for a hand (no camera required)."""
    player_cards = player.upper().split()
    counter = HiLoCounter(decks=decks)
    # Inject a synthetic running count that produces the desired true count
    counter.running_count = int(count * (decks / 2))

    state = GameState(
        player_cards=player_cards,
        dealer_upcard=dealer.upper(),
        counter=counter,
    )
    result = decide(state)

    total, is_soft = hand_total(player_cards)
    style = ACTION_STYLES.get(result["action"], "white")

    console.print()
    console.print(Panel(
        f"[{style}]{result['label']}[/{style}]",
        title=f"Hand: {' '.join(player_cards)} ({total}{'s' if is_soft else ''}) vs {dealer.upper()}",
        border_style="dim"
    ))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]True Count[/dim]",  f"[cyan]{result['true_count']:+.1f}[/cyan]")
    table.add_row("[dim]Bet Units[/dim]",   f"[yellow]{result['bet_units']}×[/yellow]")
    table.add_row("[dim]Reasoning[/dim]",   f"[dim]{result['reasoning']}[/dim]")
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# count (interactive trainer)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--decks", default=6, type=int, help="Number of decks in shoe")
def count(decks: int) -> None:
    """Interactive Hi-Lo card counting trainer. Type cards to update count."""
    counter = HiLoCounter(decks=decks)
    console.print(Panel(
        "Hi-Lo Card Counting Trainer\n"
        "Enter cards one at a time (e.g. A, K, 5, 10).\n"
        "Type [bold]reset[/bold] to start a new shoe, [bold]quit[/bold] to exit.",
        title="♠ Count Trainer", border_style="blue"
    ))
    while True:
        try:
            raw = console.input("[dim]Card > [/dim]").strip().upper()
        except (KeyboardInterrupt, EOFError):
            break
        if raw in ("Q", "QUIT", "EXIT"):
            break
        if raw in ("R", "RESET"):
            counter.reset()
            console.print("[yellow]Shoe reset.[/yellow]")
            continue
        counter.update(raw)
        tc = counter.true_count()
        colour = "green" if tc > 1 else "red" if tc < -1 else "white"
        console.print(
            f"  RC=[bold]{counter.running_count:+d}[/bold]  "
            f"TC=[{colour}]{tc:+.2f}[/{colour}]  "
            f"Cards seen: {counter.cards_seen}"
        )


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# debug-frame  — one-shot live frame diagnostic (text output, saves PNG)
# ---------------------------------------------------------------------------

@cli.command("debug-frame")
@click.option("--serial", default=None, help="ADB device serial")
@click.option("--out", default="/tmp/bj_debug_frame.png", show_default=True,
              help="Where to save the annotated debug image")
@click.option("--raw", default="/tmp/bj_raw_frame.png", show_default=True,
              help="Where to save the raw (un-annotated) frame")
@click.pass_context
def debug_frame(ctx: click.Context, serial: str, out: str, raw: str) -> None:
    """
    Grab one live ADB frame, run the detector, print a full text report,
    and save TWO PNG files:
      --raw  the plain screenshot (open in Preview to see the live state)
      --out  the annotated version with region boxes drawn on it

    No images are displayed inline — open them manually in the Finder / Preview.
    """
    import cv2
    import numpy as np
    from .game_detector import VegasBJDetector, Layout

    adb = ADBCapture(serial)
    if not adb.is_available():
        console.print("[red]No ADB device found. Connect phone via USB + USB Debugging.[/red]")
        sys.exit(1)

    console.print("[cyan]Grabbing frame from device…[/cyan]")
    frame = adb.grab()
    if frame is None:
        console.print("[red]adb screencap returned None. Check USB connection.[/red]")
        sys.exit(1)

    h, w = frame.shape[:2]
    console.print(f"  Frame size: [bold]{w}×{h}[/bold] px")

    # ── Save raw ──────────────────────────────────────────────────────────
    cv2.imwrite(raw, frame)
    console.print(f"  Raw frame : [cyan]{raw}[/cyan]")

    # ── Run detector ──────────────────────────────────────────────────────
    detector = VegasBJDetector()
    gf = detector.detect(frame)

    # ── Print text report ─────────────────────────────────────────────────
    table = Table(title="Frame Detection Report", show_header=True,
                  header_style="bold magenta")
    table.add_column("Field", style="dim", min_width=20)
    table.add_column("Value", style="bold")

    state_colour = {"playing": "green", "result": "yellow",
                    "betting": "red", "unknown": "dim"}.get(gf.game_state, "white")
    table.add_row("game_state",
                  f"[{state_colour}]{gf.game_state}[/{state_colour}]")
    table.add_row("dealer_total",    str(gf.dealer_total))
    table.add_row("player_total",    str(gf.player_total))
    table.add_row("is_soft",         str(gf.is_soft))
    table.add_row("dealer_upcard",   str(gf.dealer_upcard_rank))
    table.add_row("player_ranks",    str(gf.player_card_ranks))
    table.add_row("buttons_found",   str(list(gf.buttons.keys())))
    table.add_row("is_actionable",   str(gf.is_actionable))

    # Colour pixel counts for button strip diagnostics
    strip_y1 = int(Layout.BUTTON_ROW_Y_TOP * h)
    strip_y2 = int(Layout.BUTTON_ROW_Y_BOTTOM * h)
    strip = frame[strip_y1:strip_y2, 0:w]
    hsv_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    for btn_name, (lo, hi) in Layout.BUTTON_COLOURS.items():
        mask = cv2.inRange(hsv_strip, np.array(lo), np.array(hi))
        px = int(np.count_nonzero(mask))
        table.add_row(f"colour_{btn_name.lower()}_px", str(px))
    # The KEY discriminator: Hit bright-green (≥3000 = playing, <3000 = betting/result)
    hit_bright = cv2.inRange(hsv_strip,
                             np.array([45, 150, 150]), np.array([90, 255, 255]))
    hit_bright_px = int(np.count_nonzero(hit_bright))
    threshold_note = "✅ PLAYING" if hit_bright_px >= 3000 else "⛔ betting/result"
    table.add_row("hit_bright_green_px",
                  f"{hit_bright_px}  ← {threshold_note} (threshold=3000)")

    console.print(table)

    # ── Build annotated image (bounding boxes of the detector regions) ────
    vis = frame.copy()

    def _box(x1, y1, x2, y2, colour, label):
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    # Result region
    _box(int(Layout.RESULT_REGION_X * w),
         int(Layout.RESULT_REGION_Y * h),
         int((Layout.RESULT_REGION_X + Layout.RESULT_REGION_W) * w),
         int((Layout.RESULT_REGION_Y + Layout.RESULT_REGION_H) * h),
         (0, 255, 255), "result_rgn")

    # Dealer bubble
    dbx = int(Layout.DEALER_BUBBLE_CX * w)
    dby = int(Layout.DEALER_BUBBLE_CY * h)
    dbr = int(Layout.DEALER_BUBBLE_R  * w)
    _box(dbx - dbr, dby - dbr, dbx + dbr, dby + dbr, (255, 0, 255), "dealer_bubble")

    # Player bubble
    pbx = int(Layout.PLAYER_BUBBLE_CX * w)
    pby = int(Layout.PLAYER_BUBBLE_CY * h)
    pbr = int(Layout.PLAYER_BUBBLE_R  * w)
    _box(pbx - pbr, pby - pbr, pbx + pbr, pby + pbr, (0, 255, 0), "player_bubble")

    # Button strip
    _box(0, strip_y1, w, strip_y2, (0, 165, 255), "btn_strip")

    # Dealer card rank region
    _box(int(Layout.DEALER_CARD_RANK_X * w),
         int(Layout.DEALER_CARD_RANK_Y * h),
         int((Layout.DEALER_CARD_RANK_X + Layout.DEALER_CARD_RANK_W) * w),
         int((Layout.DEALER_CARD_RANK_Y + Layout.DEALER_CARD_RANK_H) * h),
         (255, 128, 0), "dealer_rank")

    # Player card rank region
    _box(int(Layout.PLAYER_CARD_RANK_X * w),
         int(Layout.PLAYER_CARD_RANK_Y * h),
         int((Layout.PLAYER_CARD_RANK_X + Layout.PLAYER_CARD_RANK_W) * w),
         int((Layout.PLAYER_CARD_RANK_Y + Layout.PLAYER_CARD_RANK_H) * h),
         (128, 255, 0), "player_rank")

    # Detected buttons
    for btn_name, (bx, by) in gf.buttons.items():
        cv2.drawMarker(vis, (bx, by), (0, 0, 255),
                       cv2.MARKER_CROSS, 40, 4)
        cv2.putText(vis, btn_name, (bx + 15, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(out, vis)
    console.print(f"\n  Annotated : [cyan]{out}[/cyan]")
    console.print("\n[dim]Open those PNG files in Preview/Finder to inspect.[/dim]")
    console.print("[dim]Run with -v on the CLI group for full DEBUG logs:[/dim]")
    console.print("[dim]  bj-assistant -v debug-frame[/dim]\n")


@cli.command()
@click.option("--serial", default=None, help="ADB device serial")
def calibrate(serial: str) -> None:
    """
    Interactive ADB button calibration tool.
    Captures the phone screen and helps you identify button coordinates.
    """
    adb = ADBCapture(serial)
    if not adb.is_available():
        console.print("[red]No ADB device found. Connect your phone via USB and enable USB Debugging.[/red]")
        sys.exit(1)

    console.print("[green]ADB device found![/green]")
    console.print("Capturing screen… open the BJ game on your phone now.")
    time.sleep(2)

    import cv2
    frame = adb.grab()
    if frame is None:
        console.print("[red]Failed to capture screen.[/red]")
        sys.exit(1)

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, frame)
    console.print(f"Screenshot saved to: [cyan]{tmp.name}[/cyan]")
    console.print("Open the image, note the (x, y) pixel coordinates of each button,")
    console.print("then add them to [bold]config/settings.yaml[/bold] under [bold]button_map[/bold].")

    for action in ["HIT", "STAND", "DOUBLE", "SPLIT", "SURRENDER"]:
        coords = console.input(f"  {action} button (x,y) or Enter to skip: ").strip()
        if coords:
            try:
                x, y = map(int, coords.split(","))
                console.print(f"  → {action}: ({x}, {y}) ✓")
            except ValueError:
                console.print("  [yellow]Skipped (invalid format)[/yellow]")

    console.print("\n[green]Calibration complete. Update config/settings.yaml manually.[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
