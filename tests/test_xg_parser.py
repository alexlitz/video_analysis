"""Tests for XG file parser."""

import pytest
from pathlib import Path

from src.xg_parser.xg_reader import (
    XGReader,
    XGMatch,
    XGPosition,
    validate_xg_file,
    _utf16_array_to_str,
    _delphi_short_str
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_utf16_array_to_str(self):
        # Test basic conversion
        arr = (72, 101, 108, 108, 111, 0)  # "Hello\0"
        result = _utf16_array_to_str(arr)
        assert result == "Hello"

    def test_utf16_array_empty(self):
        arr = (0,)
        result = _utf16_array_to_str(arr)
        assert result == ""

    def test_delphi_short_str(self):
        # First byte is length, rest is string
        arr = (5, 72, 101, 108, 108, 111)  # length=5, "Hello"
        result = _delphi_short_str(arr)
        assert result == "Hello"


class TestXGPosition:
    """Test XGPosition class."""

    def test_position_creation(self):
        board = tuple(range(26))
        pos = XGPosition(board=board)
        assert len(pos.board) == 26

    def test_position_to_array(self):
        board = tuple(range(26))
        pos = XGPosition(board=board)
        arr = pos.to_array()
        assert isinstance(arr, list)
        assert len(arr) == 26


class TestXGMatch:
    """Test XGMatch class."""

    def test_match_creation(self):
        match = XGMatch(
            player1="Alice",
            player2="Bob",
            match_length=11
        )
        assert match.player1 == "Alice"
        assert match.player2 == "Bob"
        assert match.match_length == 11
        assert len(match.games) == 0


class TestXGReader:
    """Test XGReader class."""

    def test_reader_initialization(self):
        reader = XGReader("dummy.xg")
        assert reader.filepath == Path("dummy.xg")
        assert reader.match is None

    def test_to_dict_empty(self):
        reader = XGReader("dummy.xg")
        result = reader.to_dict()
        assert result == {}


class TestValidation:
    """Test validation functions."""

    def test_validate_nonexistent_file(self):
        result = validate_xg_file("/nonexistent/file.xg")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
