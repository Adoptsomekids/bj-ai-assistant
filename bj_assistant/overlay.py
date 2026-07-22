"""
overlay.py
----------
Real-time HUD overlay.

Primary:  Terminal HUD — uses Rich to render a live panel in the terminal.
          Works on every platform with zero extra dependencies.

Optional: Tkinter floating window — only launched if Tkinter is importable
          and the caller passes use_tkinter=True.  Silently falls back to
          the terminal HUD if Tkinter is unavailable (e.g. Python 3.14 +
          missing Tcl/Tk on macOS Sequoia).
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Colour scheme (used by both backends)
ACTION_COLOUR_RICH = {
    "H": "bold red",
    "S": "bold green",
    "D": "bold yellow",
    "P": "bold blue",
    "R": "bold magenta",
}

ACTION_EMOJI = {
    "H": "🎯 HIT",
    "S": "✋ STAND",
    "D": "💰 DOUBLE DOWN",
    "P": "✂️  SPLIT",
    "R": "🏳️  SURRENDER",
}


# ---------------------------------------------------------------------------
# Terminal HUD (always available)
# ---------------------------------------------------------------------------

class TerminalHUD:
    """
    Renders a live Rich panel in the terminal.
    Thread-safe: update() can be called from any thread.
    """

    def __init__(self) -> None:
        self._data: Optional[dict] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start_async(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, data: dict) -> None:
        with self._lock:
            self._data = data

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        def build() -> Panel:
            with self._lock:
                data = self._data

            if data is None:
                return Panel("[dim]Waiting for cards...[/dim]",
                             title="♠ BJ AI Assistant", border_style="dim")

            action  = data.get("action", "?")
            label   = ACTION_EMOJI.get(action, data.get("label", action))
            colour  = ACTION_COLOUR_RICH.get(action, "white")
            tc      = data.get("true_count", 0.0)
            rc      = data.get("running_count", 0)
            bet     = data.get("bet_units", 1)
            total   = data.get("player_total", 0)
            soft    = data.get("is_soft", False)
            p_cards = data.get("player_cards", [])
            d_up    = data.get("dealer_upcard", "?")
            reason  = data.get("reasoning", "")

            tc_colour = "green" if tc > 1 else "red" if tc < -1 else "white"

            tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
            tbl.add_row(
                f"[{colour}]{label}[/{colour}]",
                f"[dim]You:[/dim] {' '.join(p_cards) or str(total)}{'s' if soft else ''}  "
                f"[dim]│ Dealer:[/dim] {d_up}"
            )
            tbl.add_row(
                f"[dim]True Count:[/dim] [{tc_colour}]{tc:+.1f}[/{tc_colour}]  "
                f"[dim]Running:[/dim] [{tc_colour}]{rc:+d}[/{tc_colour}]",
                f"[dim]Bet:[/dim] [yellow]{bet}×[/yellow] unit"
            )
            tbl.add_row(f"[dim]{reason}[/dim]", "")

            return Panel(tbl, title="♠ BJ AI Assistant", border_style=colour.split()[-1])

        with Live(build(), refresh_per_second=4, screen=False) as live:
            while self._running:
                live.update(build())
                time.sleep(0.25)


# ---------------------------------------------------------------------------
# Tkinter floating HUD (optional, best-effort)
# ---------------------------------------------------------------------------

def _try_import_tkinter():
    try:
        import tkinter as tk
        # Quick sanity-check: try creating a Tk root to catch macOS version errors
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return tk
    except Exception as exc:
        log.debug("Tkinter not available: %s", exc)
        return None


COLOURS = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "border":     "#30363d",
    "text":       "#e6edf3",
    "muted":      "#8b949e",
    "hit":        "#f85149",
    "stand":      "#3fb950",
    "double":     "#d29922",
    "split":      "#58a6ff",
    "surrender":  "#bc8cff",
    "count_pos":  "#3fb950",
    "count_neg":  "#f85149",
    "count_zero": "#8b949e",
}

ACTION_COLOUR_TK = {
    "H": COLOURS["hit"],
    "S": COLOURS["stand"],
    "D": COLOURS["double"],
    "P": COLOURS["split"],
    "R": COLOURS["surrender"],
}


class TkinterHUD:
    """Semi-transparent always-on-top floating window (Tkinter)."""

    def __init__(
        self,
        width: int = 420,
        height: int = 300,
        x: int = 20,
        y: int = 20,
        opacity: float = 0.92,
    ) -> None:
        self._w = width
        self._h = height
        self._x = x
        self._y = y
        self._opacity = opacity
        self._root = None
        self._pending: Optional[dict] = None
        self._lock = threading.Lock()
        self._running = False

    def start_async(self) -> threading.Thread:
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return t

    def update(self, data: dict) -> None:
        with self._lock:
            self._pending = data
        if self._root:
            self._root.after(0, self._flush)

    def stop(self) -> None:
        self._running = False
        if self._root:
            self._root.after(0, self._root.destroy)

    def _run(self) -> None:
        tk = _try_import_tkinter()
        if tk is None:
            log.warning("Tkinter unavailable — floating HUD disabled")
            return
        self._running = True
        root = tk.Tk()
        self._root = root
        root.title("BJ AI Assistant")
        root.geometry(f"{self._w}x{self._h}+{self._x}+{self._y}")
        root.configure(bg=COLOURS["bg"])
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._opacity)
        root.bind("<ButtonPress-1>", lambda e: setattr(self, '_dx', e.x) or setattr(self, '_dy', e.y))
        root.bind("<B1-Motion>", lambda e: root.geometry(
            f"+{root.winfo_x()+(e.x-getattr(self,'_dx',0))}+{root.winfo_y()+(e.y-getattr(self,'_dy',0))}"))

        tk.Label(root, text="✕", bg=COLOURS["bg"], fg=COLOURS["muted"],
                 cursor="hand2", font=("Helvetica", 13)
                 ).place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=4)

        tk.Label(root, text="♠ BJ AI Assistant", bg=COLOURS["bg"],
                 fg=COLOURS["muted"], font=("Helvetica", 11, "bold")).pack(pady=(8, 0))

        self._action_lbl = tk.Label(root, text="—", bg=COLOURS["bg"],
                                     fg=COLOURS["text"], font=("Helvetica", 36, "bold"))
        self._action_lbl.pack(pady=(6, 0))

        self._sub_lbl = tk.Label(root, text="Waiting for cards…", bg=COLOURS["bg"],
                                  fg=COLOURS["muted"], font=("Helvetica", 12), wraplength=380)
        self._sub_lbl.pack()

        tk.Frame(root, bg=COLOURS["border"], height=1).pack(fill="x", padx=16, pady=6)

        self._cards_lbl = tk.Label(root, text="", bg=COLOURS["bg"],
                                    fg=COLOURS["text"], font=("Helvetica", 12))
        self._cards_lbl.pack()

        self._count_lbl = tk.Label(root, text="", bg=COLOURS["bg"],
                                    fg=COLOURS["muted"], font=("Helvetica", 11))
        self._count_lbl.pack(pady=(4, 0))

        self._bet_lbl = tk.Label(root, text="", bg=COLOURS["bg"],
                                  fg=COLOURS["muted"], font=("Helvetica", 11))
        self._bet_lbl.pack()

        tk.Frame(root, bg=COLOURS["border"], height=1).pack(fill="x", padx=16, pady=5)

        self._reason_lbl = tk.Label(root, text="", bg=COLOURS["bg"],
                                     fg=COLOURS["muted"], font=("Helvetica", 9),
                                     wraplength=380, justify="left")
        self._reason_lbl.pack(padx=16)

        root.mainloop()

    def _flush(self) -> None:
        with self._lock:
            data = self._pending
        if data is None:
            return
        action  = data.get("action", "?")
        colour  = ACTION_COLOUR_TK.get(action, COLOURS["text"])
        label   = data.get("label", action)
        total   = data.get("player_total", 0)
        soft    = data.get("is_soft", False)
        p_cards = data.get("player_cards", [])
        d_up    = data.get("dealer_upcard", "?")
        tc      = data.get("true_count", 0.0)
        rc      = data.get("running_count", 0)
        bet     = data.get("bet_units", 1)
        tc_col  = COLOURS["count_pos"] if tc > 1 else COLOURS["count_neg"] if tc < -1 else COLOURS["count_zero"]

        self._action_lbl.configure(text=action, fg=colour)
        self._sub_lbl.configure(text=label)
        self._cards_lbl.configure(
            text=f"You: {' '.join(p_cards) or str(total)}{'s' if soft else ''}  │  Dealer: {d_up}")
        self._count_lbl.configure(
            text=f"True Count: {tc:+.1f}   Running: {rc:+d}", fg=tc_col)
        self._bet_lbl.configure(text=f"Recommended Bet: {bet}× unit")
        self._reason_lbl.configure(text=data.get("reasoning", ""))


# ---------------------------------------------------------------------------
# HUDOverlay — auto-selects backend
# ---------------------------------------------------------------------------

class HUDOverlay:
    """
    Public overlay interface. Automatically chooses:
      - TkinterHUD  if use_tkinter=True AND Tkinter is importable
      - TerminalHUD otherwise (always works)
    """

    def __init__(self, use_tkinter: bool = False) -> None:
        if use_tkinter and _try_import_tkinter() is not None:
            log.info("Using Tkinter floating HUD")
            self._impl = TkinterHUD()
        else:
            log.info("Using terminal HUD (Rich)")
            self._impl = TerminalHUD()

    def start_async(self) -> None:
        self._impl.start_async()

    def update(self, data: dict) -> None:
        self._impl.update(data)

    def stop(self) -> None:
        self._impl.stop()
