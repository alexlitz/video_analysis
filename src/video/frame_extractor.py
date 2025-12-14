"""
Frame extractor optimized for training data generation.
Uses decord for efficient video loading.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    import cv2


@dataclass
class FrameBatch:
    """A batch of video frames."""
    frames: np.ndarray  # Shape: (batch, height, width, channels)
    timestamps: np.ndarray  # Shape: (batch,)
    frame_indices: np.ndarray  # Shape: (batch,)


class FrameExtractor:
    """
    Efficient frame extraction for training data.
    Uses decord if available, falls back to OpenCV.
    """

    def __init__(
        self,
        resize: Optional[tuple[int, int]] = None,
        normalize: bool = False
    ):
        """
        Args:
            resize: Target size (width, height) or None
            normalize: Whether to normalize to [0, 1]
        """
        self.resize = resize
        self.normalize = normalize
        self.use_decord = DECORD_AVAILABLE

    def extract_frames(
        self,
        video_path: str,
        frame_indices: Optional[Union[list, np.ndarray]] = None,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None
    ) -> FrameBatch:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            frame_indices: Specific frame indices to extract
            num_frames: Number of frames to uniformly sample (if frame_indices not given)
            fps: Sample at this FPS (if frame_indices and num_frames not given)

        Returns:
            FrameBatch with extracted frames
        """
        if self.use_decord:
            return self._extract_with_decord(video_path, frame_indices, num_frames, fps)
        else:
            return self._extract_with_opencv(video_path, frame_indices, num_frames, fps)

    def _extract_with_decord(
        self,
        video_path: str,
        frame_indices: Optional[Union[list, np.ndarray]],
        num_frames: Optional[int],
        fps: Optional[float]
    ) -> FrameBatch:
        """Extract frames using decord (faster)."""
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()

        # Determine which frames to extract
        if frame_indices is not None:
            indices = np.array(frame_indices)
        elif num_frames is not None:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif fps is not None:
            step = int(video_fps / fps)
            indices = np.arange(0, total_frames, step)
        else:
            indices = np.arange(total_frames)

        # Extract frames
        frames = vr.get_batch(indices).asnumpy()

        # Resize if needed
        if self.resize is not None:
            resized = []
            for frame in frames:
                import cv2
                resized.append(cv2.resize(frame, self.resize))
            frames = np.array(resized)

        # Normalize if needed
        if self.normalize:
            frames = frames.astype(np.float32) / 255.0

        # Calculate timestamps
        timestamps = indices / video_fps

        return FrameBatch(
            frames=frames,
            timestamps=timestamps,
            frame_indices=indices
        )

    def _extract_with_opencv(
        self,
        video_path: str,
        frame_indices: Optional[Union[list, np.ndarray]],
        num_frames: Optional[int],
        fps: Optional[float]
    ) -> FrameBatch:
        """Extract frames using OpenCV (fallback)."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Determine which frames to extract
        if frame_indices is not None:
            indices = np.array(frame_indices)
        elif num_frames is not None:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif fps is not None:
            step = int(video_fps / fps)
            indices = np.arange(0, total_frames, step)
        else:
            indices = np.arange(min(total_frames, 1000))  # Limit for memory

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                frames.append(frame)

        cap.release()

        frames = np.array(frames)

        if self.normalize:
            frames = frames.astype(np.float32) / 255.0

        timestamps = indices[:len(frames)] / video_fps

        return FrameBatch(
            frames=frames,
            timestamps=timestamps,
            frame_indices=indices[:len(frames)]
        )

    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        fps: float = 1.0
    ) -> FrameBatch:
        """
        Extract frames from a time range.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            fps: Frames per second to extract

        Returns:
            FrameBatch with extracted frames
        """
        if self.use_decord:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
        else:
            import cv2
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        step = int(video_fps / fps)

        indices = list(range(start_frame, end_frame, step))

        return self.extract_frames(video_path, frame_indices=indices)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        extractor = FrameExtractor(resize=(224, 224), normalize=True)

        print(f"Using decord: {extractor.use_decord}")

        batch = extractor.extract_frames(sys.argv[1], num_frames=16)
        print(f"Extracted {len(batch.frames)} frames")
        print(f"Shape: {batch.frames.shape}")
        print(f"Timestamps: {batch.timestamps}")
    else:
        print("Usage: python frame_extractor.py <video_file>")
