"""
Tests for capture.py — backend availability checks and factory function.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from bj_assistant.capture import ADBCapture, get_best_capture


class TestADBCapture:
    def test_is_available_false_when_adb_missing(self):
        adb = ADBCapture()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert adb.is_available() is False

    def test_is_available_false_when_no_devices(self):
        adb = ADBCapture()
        mock_result = MagicMock()
        mock_result.stdout = "List of devices attached\n"
        with patch("subprocess.run", return_value=mock_result):
            assert adb.is_available() is False

    def test_tap_returns_true_on_success(self):
        adb = ADBCapture()
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert adb.tap(100, 200) is True

    def test_tap_returns_false_on_error(self):
        adb = ADBCapture()
        with patch("subprocess.run", side_effect=Exception("fail")):
            assert adb.tap(100, 200) is False


class TestGetBestCapture:
    def test_returns_adb_when_available(self):
        with patch("bj_assistant.capture.ADBCapture.is_available", return_value=True):
            capture = get_best_capture()
            assert isinstance(capture, ADBCapture)
