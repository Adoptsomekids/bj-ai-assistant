"""
overlay.py
----------
Real-time HUD overlay displayed on the Mac that shows:
  - Detected cards (player & dealer)
  - Optimal action with colour coding
  - True count and bet sizing recommendation
  - Running count

Built with Tkinter (zero external dependencies beyond stdlib) so it runs
on macOS without needing PyQt/wxPython.
"""

from __future__ import annotations
import logging
import threading
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Colour scheme
COLOURS = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "border":     "#30363d",
    "text":       "#e6edf3",
    "muted":      "#8b949e",
    "hit":        "#f85149",   # red
    "stand":      "#3fb950",   # green
    "double":     "#d29922",   # yellow
    "split":      "#58a6ff",   # blue
    "surrender":  "#bc8cff",   # purple
    "count_pos":  "#3fb950",
    "count_neg":  "#f85149",
    "count_zero": "#8b949e",
}

ACTION_COLOUR = {
    "H": COLOURS["hit"],
    "S": COLOURS["stand"],
    "D": COLOURS["double"],
    "P": COLOURS["split"],
    "R": COLOURS["surrender"],
}


class HUDOverlay:
    """
    Semi-transparent always-on-top Tkinter window that renders the BJ HUD.
    Can be updated from any thread via update().
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 320,
        x: int = 20,
        y: int = 20,
        opacity: float = 0.92,
        on_close: Optional[Callable] = None,
    ) -> None:
        self._width = width
        self._height = height
        self._x = x
        self._y = y
        self._opacity = opacity
        self._on_close = on_close
        self._root: Optional[tk.Tk] = None
        self._labels: dict[str, tk.Label] = {}
        self._running = False
        self._pending_data: Optional[dict] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the overlay in the current thread (blocking). Call from main thread."""
        self._running = True
        self._build_ui()
        self._root.mainloop()

    def start_async(self) -> threading.Thread:
        """Launch the overlay in a daemon thread."""
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t

    def update(self, data: dict) -> None:
        """
        Thread-safe update. data keys:
          action, label, reasoning, true_count, running_count,
          bet_units, player_total, is_soft,
          player_cards (list[str]), dealer_upcard (str)
        """
        with self._lock:
            self._pending_data = data
        if self._root:
            self._root.after(0, self._flush)

    def stop(self) -> None:
        self._running = False
        if self._root:
            self._root.after(0, self._root.destroy)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title("BJ AI Assistant")
        root.geometry(f"{self._width}x{self._height}+{self._x}+{self._y}")
        root.configure(bg=COLOURS["bg"])
        root.overrideredirect(True)     # borderless
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._opacity)

        # Drag to reposition
        root.bind("<ButtonPress-1>", self._drag_start)
        root.bind("<B1-Motion>", self._drag_motion)
        self._drag_x = 0
        self._drag_y = 0

        # Close button
        close_btn = tk.Label(root, text="✕", bg=COLOURS["bg"], fg=COLOURS["muted"],
                              cursor="hand2", font=("Helvetica", 13))
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=4)
        close_btn.bind("<Button-1>", lambda _: self._close())

        # Title bar
        title = tk.Label(root, text="♠ BJ AI Assistant",
                         bg=COLOURS["bg"], fg=COLOURS["muted"],
                         font=("Helvetica", 11, "bold"))
        title.pack(pady=(8, 0))

        # --- Action label (big) ---
        self._action_label = tk.Label(
            root, text="—",
            bg=COLOURS["bg"], fg=COLOURS["text"],
            font=("Helvetica", 36, "bold")
        )
        self._action_label.pack(pady=(6, 0))

        # Sub-label (full action name)
        self._action_sub = tk.Label(
            root, text="Waiting for cards...",
            bg=COLOURS["bg"], fg=COLOURS["muted"],
            font=("Helvetica", 12), wraplength=360
        )
        self._action_sub.pack()

        # Separator
        sep = tk.Frame(root, bg=COLOURS["border"], height=1)
        sep.pack(fill="x", padx=16, pady=8)

        # Cards row
        self._cards_label = tk.Label(
            root, text="",
            bg=COLOURS["bg"], fg=COLOURS["text"],
            font=("Helvetica", 13)
        )
        self._cards_label.pack()

        # Count / bet row
        self._count_label = tk.Label(
            root, text="",
            bg=COLOURS["bg"], fg=COLOURS["muted"],
            font=("Helvetica", 11)
        )
        self._count_label.pack(pady=(4, 0))

        self._bet_label = tk.Label(
            root, text="",
            bg=COLOURS["bg"], fg=COLOURS["muted"],
            font=("Helvetica", 11)
        )
        self._bet_label.pack()

        # Reasoning
        sep2 = tk.Frame(root, bg=COLOURS["border"], height=1)
        sep2.pack(fill="x", padx=16, pady=6)

        self._reason_label = tk.Label(
            root, text="",
            bg=COLOURS["bg"], fg=COLOURS["muted"],
            font=("Helvetica", 9), wraplength=360, justify="left"
        )
        self._reason_label.pack(padx=16)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        with self._lock:
            data = self._pending_data
        if data is None:
            return
        self._render(data)

    def _render(self, data: dict) -> None:
        action = data.get("action", "?")
        colour = ACTION_COLOUR.get(action, COLOURS["text"])
        label = data.get("label", action)

        self._action_label.configure(text=action, fg=colour)
        self._action_sub.configure(text=label)

        # Cards
        player_cards = data.get("player_cards", [])
        dealer_up = data.get("dealer_upcard", "?")
        total = data.get("player_total", 0)
        soft = " (soft)" if data.get("is_soft") else ""
        cards_text = (
            f"You: {' '.join(player_cards) or '?'}  [{total}{soft}]  "
            f"│  Dealer: {dealer_up}"
        )
        self._cards_label.configure(text=cards_text)

        # Count
        tc = data.get("true_count", 0.0)
        rc = data.get("running_count", 0)
        count_colour = (
            COLOURS["count_pos"] if tc > 1
            else COLOURS["count_neg"] if tc < -1
            else COLOURS["count_zero"]
        )
        self._count_label.configure(
            text=f"True Count: {tc:+.1f}   Running: {rc:+d}",
            fg=count_colour
        )

        # Bet
        bet = data.get("bet_units", 1)
        self._bet_label.configure(
            text=f"Recommended Bet: {bet}× unit",
            fg=COLOURS["count_pos"] if bet > 2 else COLOURS["muted"]
        )

        # Reasoning
        self._reason_label.configure(text=data.get("reasoning", ""))

    # ------------------------------------------------------------------
    # Drag
    # ------------------------------------------------------------------

    def _drag_start(self, event: tk.Event) -> None:
        self._drag_x = event.x
        self._drag_y = event.y

    def _drag_motion(self, event: tk.Event) -> None:
        if self._root:
            x = self._root.winfo_x() + (event.x - self._drag_x)
            y = self._root.winfo_y() + (event.y - self._drag_y)
            self._root.geometry(f"+{x}+{y}")

    def _close(self) -> None:
        if self._on_close:
            self._on_close()
        self.stop()
