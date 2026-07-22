"""
config.py
---------
Loads and validates runtime configuration from config/settings.yaml.
All values have sensible defaults so the system works out of the box.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


@dataclass
class Settings:
    # Capture
    device_serial: Optional[str] = None   # e.g. "emulator-5554"
    fps: int = 5
    capture_backend: str = "auto"          # "adb" | "scrcpy" | "macos" | "auto"

    # Strategy
    decks: int = 6
    allow_surrender: bool = True
    allow_double: bool = True
    allow_split: bool = True

    # Overlay
    show_overlay: bool = True
    overlay_x: int = 20
    overlay_y: int = 20
    overlay_opacity: float = 0.92

    # Auto-tap (use with caution)
    auto_tap: bool = False
    button_map: dict = field(default_factory=dict)

    # Logging
    log_level: str = "INFO"


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from YAML file, falling back to defaults for missing keys."""
    if not path.exists():
        log.info("No config file found at %s — using defaults", path)
        return Settings()

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    s = Settings()
    for key, value in raw.items():
        if hasattr(s, key):
            setattr(s, key, value)
        else:
            log.warning("Unknown config key: %s", key)

    return s
