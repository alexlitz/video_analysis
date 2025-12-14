"""
Video processor for backgammon match videos.
Extracts frames, detects board states, and synchronizes with XG data.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass


@dataclass
class VideoFrame:
    """A single video frame with metadata."""
    frame_number: int
    timestamp: float  # seconds
    image: np.ndarray


@dataclass
class ProcessedFrame:
    """A processed frame with extracted features."""
    frame_number: int
    timestamp: float
    image: np.ndarray
    board_detected: bool = False
    board_region: Optional[tuple] = None  # (x, y, w, h)
    dice_detected: bool = False
    dice_values: Optional[tuple] = None


class VideoProcessor:
    """Processes backgammon match videos."""

    def __init__(
        self,
        frame_interval: float = 1.0,  # Extract every N seconds
        target_resolution: tuple[int, int] = (640, 480)
    ):
        self.frame_interval = frame_interval
        self.target_resolution = target_resolution

    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """Load video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        return cap

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        cap = self.load_video(video_path)

        info = {
            "path": str(video_path),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }

        cap.release()
        return info

    def extract_frames(
        self,
        video_path: str,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> Iterator[VideoFrame]:
        """
        Extract frames from video at specified interval.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None for entire video)

        Yields:
            VideoFrame objects
        """
        cap = self.load_video(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if end_time is None:
            end_time = duration

        # Calculate frame numbers to extract
        current_time = start_time
        frame_interval_frames = int(self.frame_interval * fps)

        # Seek to start position
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_number = start_frame

        while current_time <= end_time:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if needed
            if frame.shape[1] != self.target_resolution[0] or \
               frame.shape[0] != self.target_resolution[1]:
                frame = cv2.resize(frame, self.target_resolution)

            yield VideoFrame(
                frame_number=frame_number,
                timestamp=current_time,
                image=frame
            )

            # Skip frames to match interval
            frame_number += frame_interval_frames
            current_time += self.frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        cap.release()

    def process_frame(self, frame: VideoFrame) -> ProcessedFrame:
        """
        Process a single frame to detect board and game state.

        Args:
            frame: VideoFrame to process

        Returns:
            ProcessedFrame with detection results
        """
        processed = ProcessedFrame(
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            image=frame.image
        )

        # Basic board detection using color and edge detection
        board_region = self._detect_board_region(frame.image)
        if board_region is not None:
            processed.board_detected = True
            processed.board_region = board_region

        # Dice detection (simplified)
        dice = self._detect_dice(frame.image)
        if dice:
            processed.dice_detected = True
            processed.dice_values = dice

        return processed

    def _detect_board_region(self, image: np.ndarray) -> Optional[tuple]:
        """
        Detect the backgammon board region in the image.

        Returns:
            Tuple of (x, y, width, height) or None
        """
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Look for typical backgammon board colors (brown/tan range)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 200])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest rectangular contour (likely the board)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Filter by aspect ratio (boards are roughly 2:1)
        aspect = w / h if h > 0 else 0
        if 1.5 < aspect < 2.5 and w > image.shape[1] * 0.3:
            return (x, y, w, h)

        return None

    def _detect_dice(self, image: np.ndarray) -> Optional[tuple]:
        """
        Detect dice values in the image.

        Returns:
            Tuple of dice values or None
        """
        # This is a placeholder - real implementation would use
        # a trained model or template matching
        return None

    def save_frames(
        self,
        frames: Iterator[VideoFrame],
        output_dir: str,
        prefix: str = "frame"
    ) -> list[Path]:
        """Save frames to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = []
        for frame in frames:
            filename = f"{prefix}_{frame.frame_number:06d}.jpg"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), frame.image)
            saved.append(filepath)

        return saved


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        processor = VideoProcessor(frame_interval=5.0)
        info = processor.get_video_info(sys.argv[1])
        print(f"Video info: {info}")

        # Extract a few frames
        frames = list(processor.extract_frames(sys.argv[1], end_time=30))
        print(f"Extracted {len(frames)} frames")

        for frame in frames[:3]:
            processed = processor.process_frame(frame)
            print(f"Frame {frame.frame_number}: board={processed.board_detected}")
    else:
        print("Usage: python processor.py <video_file>")
