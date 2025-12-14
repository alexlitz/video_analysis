"""
XG File Reader - Parses eXtreme Gammon match files.

Based on the XG format specification at:
https://www.extremegammon.com/xgformat.aspx

And the xgdatatools implementation by Michael Petch.
"""

import struct
import zlib
import tempfile
import os
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field
import uuid
import datetime


def _utf16_array_to_str(int_array: tuple) -> str:
    """Convert UTF-16 int array to string."""
    chars = []
    for val in int_array:
        if val == 0:
            break
        chars.append(chr(val))
    return ''.join(chars)


def _delphi_short_str(byte_array: tuple) -> str:
    """Convert Delphi short string to Python string."""
    length = byte_array[0]
    return ''.join(chr(b) for b in byte_array[1:length + 1])


def _delphi_datetime(delphi_dt: float) -> datetime.datetime:
    """Convert Delphi datetime to Python datetime."""
    delta = datetime.timedelta(
        days=int(delphi_dt),
        seconds=int(86400 * (delphi_dt % 1))
    )
    return datetime.datetime(1899, 12, 30) + delta


@dataclass
class XGPosition:
    """Backgammon board position."""
    board: tuple  # 26 values: 0-25 points from player's perspective

    def to_array(self) -> list[int]:
        """Convert to list of checker counts per point."""
        return list(self.board)

    def __repr__(self) -> str:
        return f"XGPosition({self.board})"


@dataclass
class XGMove:
    """A single move in a backgammon game."""
    player: int  # 1 or 2
    dice: tuple[int, int]
    position_before: XGPosition
    position_after: XGPosition
    moves: tuple  # Move sequence (from, dice, from, dice, ...)
    error: float = 0.0  # Error in equity
    luck: float = 0.0  # Luck value
    analysis_level: int = 0


@dataclass
class XGCubeAction:
    """A cube decision in a backgammon game."""
    player: int  # 1 or 2
    doubled: bool
    take: bool  # True if take, False if drop
    position: XGPosition
    error_double: float = 0.0
    error_take: float = 0.0


@dataclass
class XGGame:
    """A single game within a match."""
    game_number: int
    initial_score: tuple[int, int]
    crawford: bool
    initial_position: XGPosition
    moves: list = field(default_factory=list)
    cube_actions: list = field(default_factory=list)
    winner: int = 0  # 1 or -1 (player 1 or 2)
    points_won: int = 0
    termination: str = ""  # "single", "gammon", "backgammon", "resign", etc.


@dataclass
class XGMatch:
    """Complete backgammon match data."""
    player1: str
    player2: str
    match_length: int
    event: str = ""
    location: str = ""
    date: Optional[datetime.datetime] = None
    games: list[XGGame] = field(default_factory=list)
    crawford: bool = True
    jacoby: bool = False
    version: int = 0


class XGReader:
    """Reader for XG (eXtreme Gammon) match files."""

    HEADER_SIZE = 8232
    RECORD_SIZE = 2560
    MAGIC_NUMBER = b'HMGR'

    # Record types
    ENTRY_HEADER_MATCH = 0
    ENTRY_HEADER_GAME = 1
    ENTRY_CUBE = 2
    ENTRY_MOVE = 3
    ENTRY_FOOTER_GAME = 4
    ENTRY_FOOTER_MATCH = 5

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.match: Optional[XGMatch] = None

    def read(self) -> XGMatch:
        """Read and parse the XG file."""
        with open(self.filepath, 'rb') as f:
            # Read header
            header = self._read_header(f)
            if header is None:
                raise ValueError("Invalid XG file - bad header")

            # Extract and decompress archived data
            game_data = self._extract_game_data(f, header)

            # Parse game records
            self.match = self._parse_game_records(game_data)

        return self.match

    def _read_header(self, f) -> Optional[dict]:
        """Read the Game Data Format header."""
        data = f.read(self.HEADER_SIZE)
        if len(data) < self.HEADER_SIZE:
            return None

        unpacked = struct.unpack('<4BiiQiLHHBB6s1024H1024H1024H1024H', data)

        magic = bytearray(unpacked[0:4][::-1]).decode('ascii')
        if magic != 'HMGR':
            return None

        return {
            'magic': magic,
            'version': unpacked[4],
            'header_size': unpacked[5],
            'thumbnail_offset': unpacked[6],
            'thumbnail_size': unpacked[7],
            'game_name': _utf16_array_to_str(unpacked[14:1038]),
            'save_name': _utf16_array_to_str(unpacked[1038:2062]),
            'comments': _utf16_array_to_str(unpacked[3086:4110])
        }

    def _extract_game_data(self, f, header: dict) -> bytes:
        """Extract and decompress the game data from the archive."""
        # Skip to end of header
        f.seek(header['header_size'])

        # Read remaining data (compressed archive)
        compressed_data = f.read()

        # The archive has an index at the end - we need to parse it
        # For simplicity, try to decompress starting from various offsets
        for offset in range(0, min(len(compressed_data), 1000), 10):
            try:
                decompressed = zlib.decompress(compressed_data[offset:])
                if len(decompressed) > 0:
                    return decompressed
            except zlib.error:
                continue

        # If standard decompression fails, return raw data for analysis
        return compressed_data

    def _parse_game_records(self, data: bytes) -> XGMatch:
        """Parse the game records from decompressed data."""
        match = XGMatch(
            player1="Unknown",
            player2="Unknown",
            match_length=0
        )

        current_game: Optional[XGGame] = None
        offset = 0
        version = -1

        while offset + self.RECORD_SIZE <= len(data):
            record_type = data[offset + 8] if offset + 8 < len(data) else -1
            record_data = data[offset:offset + self.RECORD_SIZE]

            if record_type == self.ENTRY_HEADER_MATCH:
                match, version = self._parse_header_match(record_data)

            elif record_type == self.ENTRY_HEADER_GAME:
                current_game = self._parse_header_game(record_data, version)

            elif record_type == self.ENTRY_MOVE and current_game:
                move = self._parse_move(record_data, version)
                if move:
                    current_game.moves.append(move)

            elif record_type == self.ENTRY_CUBE and current_game:
                cube = self._parse_cube(record_data, version)
                if cube:
                    current_game.cube_actions.append(cube)

            elif record_type == self.ENTRY_FOOTER_GAME and current_game:
                self._parse_footer_game(record_data, current_game)
                match.games.append(current_game)
                current_game = None

            elif record_type == self.ENTRY_FOOTER_MATCH:
                self._parse_footer_match(record_data, match)
                break

            offset += self.RECORD_SIZE

        return match

    def _parse_header_match(self, data: bytes) -> tuple[XGMatch, int]:
        """Parse match header record."""
        try:
            unpacked = struct.unpack(
                '<9x41B41BxllBBBBddlld129Bxxxlllx',
                data[:260]
            )

            player1 = _delphi_short_str(unpacked[0:41])
            player2 = _delphi_short_str(unpacked[41:82])
            match_length = unpacked[82]
            crawford = bool(unpacked[84])
            jacoby = bool(unpacked[85])
            date_val = unpacked[92]

            match = XGMatch(
                player1=player1,
                player2=player2,
                match_length=match_length,
                crawford=crawford,
                jacoby=jacoby,
                date=_delphi_datetime(date_val) if date_val > 0 else None
            )

            # Get version from later in the record
            version = struct.unpack('<l', data[489*4:489*4+4])[0] if len(data) > 489*4+4 else 30
            match.version = version

            return match, version

        except struct.error:
            return XGMatch(player1="Unknown", player2="Unknown", match_length=0), 30

    def _parse_header_game(self, data: bytes, version: int) -> XGGame:
        """Parse game header record."""
        try:
            unpacked = struct.unpack('<9xxxxllB26bxl', data[:48])

            score1 = unpacked[0]
            score2 = unpacked[1]
            crawford = bool(unpacked[2])
            position = XGPosition(board=unpacked[3:29])
            game_number = unpacked[29]

            return XGGame(
                game_number=game_number,
                initial_score=(score1, score2),
                crawford=crawford,
                initial_position=position
            )
        except struct.error:
            return XGGame(
                game_number=0,
                initial_score=(0, 0),
                crawford=False,
                initial_position=XGPosition(board=(0,) * 26)
            )

    def _parse_move(self, data: bytes, version: int) -> Optional[XGMove]:
        """Parse move record."""
        try:
            unpacked = struct.unpack('<9x26b26bxxxl8l2ll', data[:96])

            pos_before = XGPosition(board=unpacked[0:26])
            pos_after = XGPosition(board=unpacked[26:52])
            player = unpacked[52]
            moves = unpacked[53:61]
            dice = unpacked[61:63]

            return XGMove(
                player=player,
                dice=(dice[0], dice[1]),
                position_before=pos_before,
                position_after=pos_after,
                moves=moves
            )
        except struct.error:
            return None

    def _parse_cube(self, data: bytes, version: int) -> Optional[XGCubeAction]:
        """Parse cube action record."""
        try:
            unpacked = struct.unpack('<9xxxxllllll26b', data[:64])

            player = unpacked[0]
            doubled = bool(unpacked[1])
            take = bool(unpacked[2])
            position = XGPosition(board=unpacked[6:32])

            return XGCubeAction(
                player=player,
                doubled=doubled,
                take=take,
                position=position
            )
        except struct.error:
            return None

    def _parse_footer_game(self, data: bytes, game: XGGame) -> None:
        """Parse game footer record."""
        try:
            unpacked = struct.unpack('<9xxxxllBxxxlll', data[:32])

            game.winner = unpacked[3]
            game.points_won = unpacked[4]

            termination = unpacked[5]
            term_map = {0: "drop", 1: "single", 2: "gammon", 3: "backgammon"}
            if termination >= 100:
                game.termination = "resign_" + term_map.get(termination - 100, "unknown")
            elif termination >= 1000:
                game.termination = "settle_" + term_map.get(termination - 1000, "unknown")
            else:
                game.termination = term_map.get(termination, "unknown")

        except struct.error:
            pass

    def _parse_footer_match(self, data: bytes, match: XGMatch) -> None:
        """Parse match footer record."""
        # Most important data already captured
        pass

    def to_dict(self) -> dict:
        """Convert match to dictionary format."""
        if not self.match:
            return {}

        return {
            'player1': self.match.player1,
            'player2': self.match.player2,
            'match_length': self.match.match_length,
            'event': self.match.event,
            'location': self.match.location,
            'date': str(self.match.date) if self.match.date else None,
            'crawford': self.match.crawford,
            'jacoby': self.match.jacoby,
            'games': [
                {
                    'game_number': g.game_number,
                    'initial_score': g.initial_score,
                    'crawford': g.crawford,
                    'winner': g.winner,
                    'points_won': g.points_won,
                    'termination': g.termination,
                    'num_moves': len(g.moves),
                    'num_cube_actions': len(g.cube_actions)
                }
                for g in self.match.games
            ]
        }


def validate_xg_file(filepath: str) -> bool:
    """Check if a file is a valid XG file."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'RGMH'
    except:
        return False


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        reader = XGReader(sys.argv[1])
        match = reader.read()
        print(json.dumps(reader.to_dict(), indent=2))
    else:
        print("Usage: python xg_reader.py <xg_file>")
