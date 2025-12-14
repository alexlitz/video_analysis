"""
YouTube scraper for BackgammonCafe channel.
Extracts video metadata and attempts to find associated XG file links.
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """Metadata for a single video."""
    video_id: str
    title: str
    description: str
    upload_date: str
    duration: int
    channel_url: str
    webpage_url: str

    # Parsed match info
    player1: Optional[str] = None
    player2: Optional[str] = None
    event: Optional[str] = None
    round_info: Optional[str] = None
    match_length: Optional[int] = None

    # Associated XG file (if found)
    xg_file_url: Optional[str] = None


class BackgammonCafeScraper:
    """Scraper for BackgammonCafe YouTube channel."""

    CHANNEL_URL = "https://www.youtube.com/@BackgammonCafe"
    CHANNEL_VIDEOS_URL = f"{CHANNEL_URL}/videos"

    # Common patterns in video titles
    MATCH_PATTERN = re.compile(
        r"(?P<year>\d{4})?\s*"
        r"(?P<event>[\w\s]+?)?\s*"
        r"(?P<player1>[\w\s\.]+?)\s+(?:vs?\.?|VS)\s+(?P<player2>[\w\s\.]+?)\s*"
        r"(?:Rd\.?\s*(?P<round>\d+))?\s*"
        r"(?:(?P<length>\d+)\s*pts?)?"
    )

    def __init__(self, output_dir: str = "data/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_channel_videos(self, limit: Optional[int] = None) -> list[VideoMetadata]:
        """
        Fetch all video metadata from the channel.

        Args:
            limit: Maximum number of videos to fetch (None for all)

        Returns:
            List of VideoMetadata objects
        """
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "-J",  # Output as JSON
            self.CHANNEL_VIDEOS_URL
        ]

        if limit:
            cmd.extend(["--playlist-end", str(limit)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        playlist_data = json.loads(result.stdout)
        videos = []

        for entry in playlist_data.get("entries", []):
            metadata = self._fetch_video_metadata(entry["id"])
            if metadata:
                videos.append(metadata)

        return videos

    def _fetch_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Fetch detailed metadata for a single video."""
        cmd = [
            "yt-dlp",
            "-J",
            "--skip-download",
            f"https://www.youtube.com/watch?v={video_id}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Warning: Could not fetch metadata for {video_id}")
            return None

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        metadata = VideoMetadata(
            video_id=video_id,
            title=data.get("title", ""),
            description=data.get("description", ""),
            upload_date=data.get("upload_date", ""),
            duration=data.get("duration", 0),
            channel_url=data.get("channel_url", ""),
            webpage_url=data.get("webpage_url", f"https://www.youtube.com/watch?v={video_id}")
        )

        # Parse match info from title
        self._parse_match_info(metadata)

        # Look for XG file links in description
        self._find_xg_links(metadata)

        return metadata

    def _parse_match_info(self, metadata: VideoMetadata) -> None:
        """Extract match information from video title."""
        match = self.MATCH_PATTERN.search(metadata.title)
        if match:
            groups = match.groupdict()
            metadata.player1 = groups.get("player1", "").strip() if groups.get("player1") else None
            metadata.player2 = groups.get("player2", "").strip() if groups.get("player2") else None
            metadata.event = groups.get("event", "").strip() if groups.get("event") else None
            metadata.round_info = groups.get("round") if groups.get("round") else None
            metadata.match_length = int(groups["length"]) if groups.get("length") else None

    def _find_xg_links(self, metadata: VideoMetadata) -> None:
        """Search description for XG file download links."""
        description = metadata.description

        # Common patterns for file sharing links
        patterns = [
            r'(https?://drive\.google\.com/[^\s]+)',
            r'(https?://www\.dropbox\.com/[^\s]+)',
            r'(https?://[^\s]+\.xg)',
            r'(https?://[^\s]+xg[^\s]*\.zip)',
            r'(https?://mega\.nz/[^\s]+)',
            r'(https?://thebackgammoncafe\.com/[^\s]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                metadata.xg_file_url = match.group(1)
                break

    def save_metadata(self, videos: list[VideoMetadata], filename: str = "video_metadata.json") -> None:
        """Save video metadata to JSON file."""
        output_path = self.output_dir / filename

        data = [
            {
                "video_id": v.video_id,
                "title": v.title,
                "description": v.description,
                "upload_date": v.upload_date,
                "duration": v.duration,
                "channel_url": v.channel_url,
                "webpage_url": v.webpage_url,
                "player1": v.player1,
                "player2": v.player2,
                "event": v.event,
                "round_info": v.round_info,
                "match_length": v.match_length,
                "xg_file_url": v.xg_file_url
            }
            for v in videos
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved metadata for {len(videos)} videos to {output_path}")

    def load_metadata(self, filename: str = "video_metadata.json") -> list[VideoMetadata]:
        """Load video metadata from JSON file."""
        input_path = self.output_dir / filename

        with open(input_path, "r") as f:
            data = json.load(f)

        return [VideoMetadata(**item) for item in data]


if __name__ == "__main__":
    scraper = BackgammonCafeScraper()

    print("Fetching videos from BackgammonCafe channel...")
    videos = scraper.get_channel_videos(limit=10)

    print(f"\nFound {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.title}")
        if v.player1 and v.player2:
            print(f"    Players: {v.player1} vs {v.player2}")
        if v.xg_file_url:
            print(f"    XG File: {v.xg_file_url}")

    scraper.save_metadata(videos)
