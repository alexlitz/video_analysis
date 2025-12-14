"""
Training loop for backgammon video analysis model.
"""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .video_model import VideoBackgammonModel, BackgammonLoss
from .dataset import BackgammonDataset


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    backbone: str = "resnet18"
    hidden_dim: int = 512
    num_frames: int = 16
    dropout: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Data
    video_dir: str = "data/videos"
    xg_dir: str = "data/xg_files"
    frame_size: tuple = (224, 224)
    fps: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_every: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Trainer:
    """Training manager for backgammon video model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = VideoBackgammonModel(
            backbone=config.backbone,
            hidden_dim=config.hidden_dim,
            num_frames=config.num_frames,
            dropout=config.dropout
        ).to(self.device)

        # Loss function
        self.criterion = BackgammonLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate / 100
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

    def create_dataloaders(
        self,
        train_manifest: Optional[str] = None,
        val_manifest: Optional[str] = None
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation dataloaders."""

        train_dataset = BackgammonDataset(
            video_dir=self.config.video_dir,
            xg_dir=self.config.xg_dir,
            manifest_path=train_manifest,
            clip_length=self.config.num_frames,
            frame_size=self.config.frame_size,
            fps=self.config.fps
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        val_loader = None
        if val_manifest:
            val_dataset = BackgammonDataset(
                video_dir=self.config.video_dir,
                xg_dir=self.config.xg_dir,
                manifest_path=val_manifest,
                clip_length=self.config.num_frames,
                frame_size=self.config.frame_size,
                fps=self.config.fps
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        return train_loader, val_loader

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            frames = batch["frames"].to(self.device)
            targets = {"position": batch["target"].to(self.device)}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(frames)
            losses = self.criterion(predictions, targets)
            loss = losses["total"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % self.config.log_every == 0:
                self.history["train_loss"].append(loss.item())
                self.history["learning_rate"].append(
                    self.optimizer.param_groups[0]["lr"]
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validation"):
            frames = batch["frames"].to(self.device)
            targets = {"position": batch["target"].to(self.device)}

            predictions = self.model(frames)
            losses = self.criterion(predictions, targets)
            loss = losses["total"]

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history["val_loss"].append(avg_loss)

        return avg_loss

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> Path:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
            "history": self.history
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint.get("history", self.history)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> None:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    print(f"New best model saved!")

            # Learning rate scheduling
            self.scheduler.step()

            # Regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        # Save training history
        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print("Training complete!")


def train_model(
    video_dir: str = "data/videos",
    xg_dir: str = "data/xg_files",
    output_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4
) -> None:
    """
    Convenience function to train the model.

    Args:
        video_dir: Directory containing training videos
        xg_dir: Directory containing XG files
        output_dir: Directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    config = TrainingConfig(
        video_dir=video_dir,
        xg_dir=xg_dir,
        checkpoint_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    trainer = Trainer(config)
    train_loader, val_loader = trainer.create_dataloaders()

    if len(train_loader.dataset) == 0:
        print("No training data found! Please add videos and XG files.")
        return

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train backgammon video model")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--xg-dir", default="data/xg_files")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    train_model(
        video_dir=args.video_dir,
        xg_dir=args.xg_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
