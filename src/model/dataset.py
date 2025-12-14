"""
PyTorch Dataset for backgammon video-to-XG prediction.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset

from ..video.frame_extractor import FrameExtractor
from ..xg_parser.xg_reader import XGReader, XGMatch, XGPosition


class BackgammonDataset(Dataset):
    """
    Dataset for training video-to-board-state models.

    Each sample contains:
    - Video frames from a segment of a match
    - Corresponding board positions from XG file
    - Move information (dice, player, etc.)
    """

    def __init__(
        self,
        video_dir: str,
        xg_dir: str,
        manifest_path: Optional[str] = None,
        clip_length: int = 16,
        frame_size: tuple[int, int] = (224, 224),
        fps: float = 1.0,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            video_dir: Directory containing video files
            xg_dir: Directory containing XG files
            manifest_path: JSON file mapping videos to XG files
            clip_length: Number of frames per clip
            frame_size: Target frame size (width, height)
            fps: Frames per second to extract
            transform: Optional transform to apply to frames
        """
        self.video_dir = Path(video_dir)
        self.xg_dir = Path(xg_dir)
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.fps = fps
        self.transform = transform

        self.frame_extractor = FrameExtractor(resize=frame_size, normalize=True)

        # Load or generate manifest
        if manifest_path and Path(manifest_path).exists():
            self.manifest = self._load_manifest(manifest_path)
        else:
            self.manifest = self._generate_manifest()

        # Build sample index
        self.samples = self._build_sample_index()

    def _load_manifest(self, path: str) -> list[dict]:
        """Load manifest mapping videos to XG files."""
        with open(path, 'r') as f:
            return json.load(f)

    def _generate_manifest(self) -> list[dict]:
        """Auto-generate manifest from directory contents."""
        manifest = []

        videos = list(self.video_dir.glob("*.mp4")) + list(self.video_dir.glob("*.webm"))
        xg_files = list(self.xg_dir.glob("*.xg"))

        # Try to match by filename similarity
        for video in videos:
            video_stem = video.stem.lower()

            # Find best matching XG file
            best_match = None
            best_score = 0

            for xg in xg_files:
                xg_stem = xg.stem.lower()
                # Simple word overlap scoring
                video_words = set(video_stem.split('_'))
                xg_words = set(xg_stem.split('_'))
                score = len(video_words & xg_words)

                if score > best_score:
                    best_score = score
                    best_match = xg

            if best_match:
                manifest.append({
                    "video": str(video),
                    "xg_file": str(best_match),
                    "time_offset": 0  # Manual adjustment needed
                })

        return manifest

    def _build_sample_index(self) -> list[dict]:
        """Build index of all training samples."""
        samples = []

        for entry in self.manifest:
            video_path = entry["video"]
            xg_path = entry.get("xg_file")

            if not Path(video_path).exists():
                continue

            # Load XG file to get game structure
            xg_data = None
            if xg_path and Path(xg_path).exists():
                try:
                    reader = XGReader(xg_path)
                    xg_data = reader.read()
                except Exception as e:
                    print(f"Warning: Could not read {xg_path}: {e}")

            # Get video duration
            try:
                from ..video.processor import VideoProcessor
                processor = VideoProcessor()
                info = processor.get_video_info(video_path)
                duration = info["duration"]
            except:
                duration = 3600  # Default 1 hour

            # Create samples at regular intervals
            time_offset = entry.get("time_offset", 0)
            clip_duration = self.clip_length / self.fps

            current_time = time_offset
            move_idx = 0

            while current_time + clip_duration < duration:
                sample = {
                    "video_path": video_path,
                    "start_time": current_time,
                    "end_time": current_time + clip_duration
                }

                # Try to associate with XG move
                if xg_data and xg_data.games:
                    total_moves = sum(len(g.moves) for g in xg_data.games)
                    if move_idx < total_moves:
                        sample["move_idx"] = move_idx
                        sample["xg_data"] = xg_data
                        move_idx += 1

                samples.append(sample)
                current_time += clip_duration

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Extract video frames
        batch = self.frame_extractor.extract_clip(
            sample["video_path"],
            sample["start_time"],
            sample["end_time"],
            fps=self.fps
        )

        frames = batch.frames

        # Pad or truncate to clip_length
        if len(frames) < self.clip_length:
            padding = np.zeros(
                (self.clip_length - len(frames),) + frames.shape[1:],
                dtype=frames.dtype
            )
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > self.clip_length:
            frames = frames[:self.clip_length]

        # Convert to tensor (C, T, H, W) format
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        if self.transform:
            frames = self.transform(frames)

        # Build target
        target = self._get_target(sample)

        return {
            "frames": frames,
            "target": target,
            "metadata": {
                "video_path": sample["video_path"],
                "start_time": sample["start_time"]
            }
        }

    def _get_target(self, sample: dict) -> torch.Tensor:
        """
        Build target tensor from XG data.

        The target encodes the board position as a 28-dimensional vector:
        - 24 points (positive for player 1 checkers, negative for player 2)
        - 2 bar positions
        - 2 borne off counts
        """
        if "xg_data" not in sample or "move_idx" not in sample:
            # Return zero target if no XG data
            return torch.zeros(28, dtype=torch.float32)

        xg_data = sample["xg_data"]
        move_idx = sample["move_idx"]

        # Find the move
        current_idx = 0
        for game in xg_data.games:
            for move in game.moves:
                if current_idx == move_idx:
                    # Convert position to target tensor
                    pos = move.position_before.board
                    target = torch.tensor(pos[:26], dtype=torch.float32)
                    # Add bar and borne off (simplified)
                    target = torch.cat([
                        target,
                        torch.zeros(2, dtype=torch.float32)  # Placeholder
                    ])
                    return target
                current_idx += 1

        return torch.zeros(28, dtype=torch.float32)

    def save_manifest(self, path: str) -> None:
        """Save manifest to file."""
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2)


class BoardPositionDataset(Dataset):
    """
    Dataset for board position recognition training.
    Uses extracted frames paired with known positions.
    """

    def __init__(
        self,
        frames_dir: str,
        positions_file: str,
        transform: Optional[Callable] = None
    ):
        self.frames_dir = Path(frames_dir)
        self.transform = transform

        with open(positions_file, 'r') as f:
            self.positions = json.load(f)

        self.frame_files = sorted(self.frames_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return min(len(self.frame_files), len(self.positions))

    def __getitem__(self, idx: int) -> dict:
        import cv2

        # Load frame
        frame = cv2.imread(str(self.frame_files[idx]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0

        # Convert to tensor
        frame = torch.from_numpy(frame).permute(2, 0, 1)

        if self.transform:
            frame = self.transform(frame)

        # Get position
        position = torch.tensor(self.positions[idx], dtype=torch.float32)

        return {"frame": frame, "position": position}


if __name__ == "__main__":
    # Test dataset creation
    dataset = BackgammonDataset(
        video_dir="data/videos",
        xg_dir="data/xg_files",
        clip_length=16,
        fps=1.0
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Frame shape: {sample['frames'].shape}")
        print(f"Target shape: {sample['target'].shape}")
