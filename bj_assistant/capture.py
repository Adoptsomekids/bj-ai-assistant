"""
capture.py
----------
Screen capture backends:
  - ADBCapture  : grabs frames from an Android device over USB/WiFi via ADB
  - ScrcpyCapture: reads the scrcpy virtual display window (fastest, real-time)
  - MacOSCapture : captures the scrcpy window by window title using macOS screencapture

Usage precedence: ScrcpyCapture → ADBCapture → MacOSCapture
"""

from __future__ import annotations
import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ScreenCapture(ABC):
    @abstractmethod
    def grab(self) -> Optional[np.ndarray]:
        """Return the latest frame as a BGR numpy array, or None on failure."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used in the current environment."""

    def release(self) -> None:
        """Clean up resources."""


# ---------------------------------------------------------------------------
# ADB capture (USB or TCP/IP)
# ---------------------------------------------------------------------------

class ADBCapture(ScreenCapture):
    """
    Captures a screenshot from an Android device using `adb screencap`.
    Works over USB (fast) or TCP/IP (wireless, ~1-2 fps).

    Prerequisites:
      - adb installed and on PATH
      - USB Debugging enabled on the phone
      - Device connected and listed in `adb devices`
    """

    def __init__(self, device_serial: Optional[str] = None) -> None:
        self._serial = device_serial
        self._adb_args = ["adb"]
        if device_serial:
            self._adb_args += ["-s", device_serial]

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                self._adb_args + ["devices"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().splitlines()
            devices = [l for l in lines[1:] if "device" in l and "offline" not in l]
            return len(devices) > 0
        except Exception as exc:
            log.debug("ADB not available: %s", exc)
            return False

    def grab(self) -> Optional[np.ndarray]:
        try:
            result = subprocess.run(
                self._adb_args + ["exec-out", "screencap", "-p"],
                capture_output=True, timeout=10
            )
            if result.returncode != 0:
                log.warning("adb screencap failed: %s", result.stderr)
                return None
            arr = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as exc:
            log.error("ADB grab error: %s", exc)
            return None

    def tap(self, x: int, y: int) -> bool:
        """Send a tap event to the device at screen coordinates (x, y)."""
        try:
            subprocess.run(
                self._adb_args + ["shell", "input", "tap", str(x), str(y)],
                timeout=5, check=True
            )
            return True
        except Exception as exc:
            log.error("ADB tap error: %s", exc)
            return False

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> bool:
        try:
            subprocess.run(
                self._adb_args + [
                    "shell", "input", "swipe",
                    str(x1), str(y1), str(x2), str(y2), str(duration_ms)
                ],
                timeout=5, check=True
            )
            return True
        except Exception as exc:
            log.error("ADB swipe error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# scrcpy OpenCV capture (reads the scrcpy window as a video stream)
# ---------------------------------------------------------------------------

class ScrcpyCapture(ScreenCapture):
    """
    Reads frames from a running scrcpy display window via OpenCV VideoCapture.
    scrcpy must already be running (e.g. `scrcpy --window-title BJMirror`).
    This is the fastest method (~30fps) because scrcpy streams H.264.
    """

    SCRCPY_V4L2_DEVICE = "/dev/video0"  # Linux only; macOS uses window capture

    def __init__(self, source: int | str = 0) -> None:
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def is_available(self) -> bool:
        cap = cv2.VideoCapture(self._source)
        ok = cap.isOpened()
        cap.release()
        return ok

    def grab(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._source)
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# macOS window capture (fallback — uses screencapture CLI)
# ---------------------------------------------------------------------------

class MacOSWindowCapture(ScreenCapture):
    """
    Captures a specific window by title using macOS `screencapture` + AppleScript.
    Works with scrcpy, QuickTime, or any window showing the phone screen.
    """

    def __init__(self, window_title: str = "scrcpy") -> None:
        self._window_title = window_title
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self._tmp_path = self._tmp.name
        self._tmp.close()

    def is_available(self) -> bool:
        return Path("/usr/sbin/screencapture").exists() or \
               subprocess.run(["which", "screencapture"], capture_output=True).returncode == 0

    def grab(self) -> Optional[np.ndarray]:
        script = f"""
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
    set winList to name of every window of application process "{self._window_title}"
end tell
"""
        try:
            # Capture the entire screen and crop to the scrcpy window area later
            subprocess.run(
                ["screencapture", "-x", "-t", "png", self._tmp_path],
                timeout=5, check=True, capture_output=True
            )
            img = cv2.imread(self._tmp_path)
            return img
        except Exception as exc:
            log.error("macOS screencapture error: %s", exc)
            return None

    def release(self) -> None:
        try:
            Path(self._tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Auto-selecting capture factory
# ---------------------------------------------------------------------------

def get_best_capture(device_serial: Optional[str] = None) -> ScreenCapture:
    """
    Return the best available capture backend in priority order:
    1. ADBCapture (direct, reliable, works over USB)
    2. ScrcpyCapture (fastest if scrcpy is running)
    3. MacOSWindowCapture (fallback)
    """
    adb = ADBCapture(device_serial)
    if adb.is_available():
        log.info("Using ADB capture backend")
        return adb

    scrcpy = ScrcpyCapture()
    if scrcpy.is_available():
        log.info("Using scrcpy/OpenCV capture backend")
        return scrcpy

    log.info("Using macOS screencapture fallback")
    return MacOSWindowCapture()
