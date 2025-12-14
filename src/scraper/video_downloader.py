"""
Video downloader for BackgammonCafe videos.
Downloads videos at specified quality for training.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .youtube_scraper import VideoMetadata


class VideoDownloader:
    """Downloads videos from YouTube for training."""

    def __init__(
        self,
        output_dir: str = "data/videos",
        quality: str = "720p",
        format_type: str = "mp4"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.format_type = format_type

    def download_video(
        self,
        video: VideoMetadata,
        output_filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download a single video.

        Args:
            video: VideoMetadata object
            output_filename: Custom filename (without extension)

        Returns:
            Path to downloaded file or None if failed
        """
        if output_filename is None:
            output_filename = f"{video.video_id}"

        output_path = self.output_dir / f"{output_filename}.{self.format_type}"

        # Skip if already downloaded
        if output_path.exists():
            print(f"Already downloaded: {output_filename}")
            return output_path

        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "-f", self._get_format_string(),
            "-o", str(output_path),
            "--no-playlist",
            video.webpage_url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                print(f"Failed to download {video.video_id}: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"Timeout downloading {video.video_id}")
            return None
        except Exception as e:
            print(f"Error downloading {video.video_id}: {e}")
            return None

    def _get_format_string(self) -> str:
        """Get yt-dlp format string based on quality setting."""
        quality_map = {
            "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "best": "bestvideo+bestaudio/best"
        }
        return quality_map.get(self.quality, quality_map["720p"])

    def download_batch(
        self,
        videos: list[VideoMetadata],
        max_workers: int = 4,
        progress_bar: bool = True
    ) -> dict[str, Optional[Path]]:
        """
        Download multiple videos in parallel.

        Args:
            videos: List of VideoMetadata objects
            max_workers: Number of parallel downloads
            progress_bar: Show progress bar

        Returns:
            Dictionary mapping video_id to downloaded path (or None)
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_video, video): video
                for video in videos
            }

            iterator = as_completed(futures)
            if progress_bar:
                iterator = tqdm(iterator, total=len(videos), desc="Downloading")

            for future in iterator:
                video = futures[future]
                try:
                    path = future.result()
                    results[video.video_id] = path
                except Exception as e:
                    print(f"Error with {video.video_id}: {e}")
                    results[video.video_id] = None

        return results

    def get_video_info(self, video_id: str) -> Optional[dict]:
        """Get video information without downloading."""
        cmd = [
            "yt-dlp",
            "-J",
            "--skip-download",
            f"https://www.youtube.com/watch?v={video_id}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return json.loads(result.stdout)
        return None


if __name__ == "__main__":
    from .youtube_scraper import BackgammonCafeScraper

    # Example usage
    scraper = BackgammonCafeScraper()
    downloader = VideoDownloader(quality="720p")

    # Get some videos
    videos = scraper.get_channel_videos(limit=2)

    # Download them
    results = downloader.download_batch(videos, max_workers=2)

    for video_id, path in results.items():
        if path:
            print(f"Downloaded {video_id} to {path}")
        else:
            print(f"Failed to download {video_id}")
