"""Tests for video model."""

import pytest
import torch

from src.model.video_model import (
    VideoBackgammonModel,
    BackgammonLoss,
    PositionalEncoding
)


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_encoding_shape(self):
        pe = PositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(2, 16, 256)  # batch, seq_len, dim
        output = pe(x)
        assert output.shape == x.shape

    def test_encoding_adds_position(self):
        pe = PositionalEncoding(d_model=256, max_len=100)
        x = torch.zeros(1, 10, 256)
        output = pe(x)
        # Output should not be all zeros (position encoding added)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestVideoBackgammonModel:
    """Test video model."""

    @pytest.fixture
    def model(self):
        return VideoBackgammonModel(
            backbone="simple",
            hidden_dim=256,
            num_frames=8,
            dropout=0.0
        )

    def test_model_forward(self, model):
        batch_size = 2
        num_frames = 8
        x = torch.randn(batch_size, 3, num_frames, 224, 224)

        output = model(x)

        assert "position" in output
        assert "dice" in output
        assert "cube" in output

        assert output["position"].shape == (batch_size, 28)
        assert output["dice"].shape == (batch_size, 2, 6)
        assert output["cube"].shape == (batch_size, 7)

    def test_model_with_features(self, model):
        x = torch.randn(1, 3, 8, 224, 224)
        output = model(x, return_features=True)

        assert "features" in output

    def test_model_different_backbones(self):
        # Test simple backbone
        model_simple = VideoBackgammonModel(backbone="simple", hidden_dim=128)
        x = torch.randn(1, 3, 8, 224, 224)
        output = model_simple(x)
        assert output["position"].shape == (1, 28)


class TestBackgammonLoss:
    """Test loss function."""

    @pytest.fixture
    def loss_fn(self):
        return BackgammonLoss()

    def test_position_loss(self, loss_fn):
        predictions = {"position": torch.randn(2, 28)}
        targets = {"position": torch.randn(2, 28)}

        losses = loss_fn(predictions, targets)

        assert "position" in losses
        assert "total" in losses
        assert losses["total"].item() > 0

    def test_all_losses(self, loss_fn):
        batch_size = 2
        predictions = {
            "position": torch.randn(batch_size, 28),
            "dice": torch.randn(batch_size, 2, 6),
            "cube": torch.randn(batch_size, 7)
        }
        targets = {
            "position": torch.randn(batch_size, 28),
            "dice": torch.randint(0, 6, (batch_size, 2)),
            "cube": torch.randint(0, 7, (batch_size,))
        }

        losses = loss_fn(predictions, targets)

        assert "position" in losses
        assert "dice" in losses
        assert "cube" in losses
        assert "total" in losses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
