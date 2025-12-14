"""
Video-to-board-state model for backgammon.

This model takes video frames as input and predicts:
1. Board position (checker locations)
2. Dice values
3. Cube state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class VideoBackgammonModel(nn.Module):
    """
    Model for predicting backgammon game state from video.

    Architecture:
    1. Frame encoder (CNN backbone)
    2. Temporal aggregation (Transformer or LSTM)
    3. State prediction heads
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        hidden_dim: int = 512,
        num_frames: int = 16,
        num_positions: int = 28,  # 24 points + 2 bar + 2 borne off
        num_dice_classes: int = 6,  # 1-6
        dropout: float = 0.1,
        use_transformer: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.use_transformer = use_transformer

        # Frame encoder (CNN backbone)
        self.frame_encoder = self._create_backbone(backbone)

        # Temporal modeling
        if use_transformer:
            self.pos_encoding = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        else:
            self.temporal_encoder = nn.LSTM(
                hidden_dim, hidden_dim, num_layers=2,
                batch_first=True, bidirectional=True, dropout=dropout
            )
            self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Prediction heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_positions)
        )

        self.dice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_dice_classes * 2)  # Two dice
        )

        self.cube_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 7)  # Cube value: 1,2,4,8,16,32,64
        )

    def _create_backbone(self, backbone: str) -> nn.Module:
        """Create CNN backbone for frame encoding."""
        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Remove final FC layer
            layers = list(model.children())[:-1]
            encoder = nn.Sequential(*layers)
            # Add projection to hidden_dim
            return nn.Sequential(
                encoder,
                nn.Flatten(),
                nn.Linear(512, self.hidden_dim)
            )

        elif backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            layers = list(model.children())[:-1]
            encoder = nn.Sequential(*layers)
            return nn.Sequential(
                encoder,
                nn.Flatten(),
                nn.Linear(2048, self.hidden_dim)
            )

        elif backbone == "efficientnet":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            model.classifier = nn.Linear(1280, self.hidden_dim)
            return model

        else:
            # Simple CNN
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, self.hidden_dim)
            )

    def forward(
        self,
        frames: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            frames: Input frames of shape (B, C, T, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with predictions
        """
        B, C, T, H, W = frames.shape

        # Encode each frame
        frames_flat = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frame_features = self.frame_encoder(frames_flat)
        frame_features = frame_features.reshape(B, T, self.hidden_dim)

        # Temporal modeling
        if self.use_transformer:
            frame_features = self.pos_encoding(frame_features)
            temporal_features = self.temporal_encoder(frame_features)
        else:
            temporal_features, _ = self.temporal_encoder(frame_features)
            temporal_features = self.lstm_proj(temporal_features)

        # Global pooling over time
        pooled = temporal_features.mean(dim=1)

        # Predictions
        position_pred = self.position_head(pooled)
        dice_pred = self.dice_head(pooled).reshape(B, 2, 6)
        cube_pred = self.cube_head(pooled)

        output = {
            "position": position_pred,  # (B, 28)
            "dice": dice_pred,  # (B, 2, 6) - logits for each die
            "cube": cube_pred,  # (B, 7) - cube value logits
        }

        if return_features:
            output["features"] = temporal_features

        return output


class BackgammonLoss(nn.Module):
    """Combined loss for backgammon prediction."""

    def __init__(
        self,
        position_weight: float = 1.0,
        dice_weight: float = 0.5,
        cube_weight: float = 0.2
    ):
        super().__init__()
        self.position_weight = position_weight
        self.dice_weight = dice_weight
        self.cube_weight = cube_weight

        self.position_loss = nn.MSELoss()
        self.dice_loss = nn.CrossEntropyLoss()
        self.cube_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: dict,
        targets: dict
    ) -> dict:
        """
        Compute losses.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary with individual and total loss
        """
        losses = {}

        # Position loss (regression)
        if "position" in targets:
            losses["position"] = self.position_loss(
                predictions["position"],
                targets["position"]
            ) * self.position_weight

        # Dice loss (classification)
        if "dice" in targets:
            dice_pred = predictions["dice"].reshape(-1, 6)
            dice_target = targets["dice"].reshape(-1)
            losses["dice"] = self.dice_loss(dice_pred, dice_target) * self.dice_weight

        # Cube loss (classification)
        if "cube" in targets:
            losses["cube"] = self.cube_loss(
                predictions["cube"],
                targets["cube"]
            ) * self.cube_weight

        # Total loss
        losses["total"] = sum(losses.values())

        return losses


if __name__ == "__main__":
    # Test model
    model = VideoBackgammonModel(backbone="simple", hidden_dim=256)

    # Create dummy input
    batch_size = 2
    num_frames = 16
    height, width = 224, 224

    x = torch.randn(batch_size, 3, num_frames, height, width)

    # Forward pass
    output = model(x)

    print(f"Position shape: {output['position'].shape}")
    print(f"Dice shape: {output['dice'].shape}")
    print(f"Cube shape: {output['cube'].shape}")

    # Test loss
    loss_fn = BackgammonLoss()
    targets = {
        "position": torch.randn(batch_size, 28)
    }
    losses = loss_fn(output, targets)
    print(f"Total loss: {losses['total'].item():.4f}")
