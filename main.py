#!/usr/bin/env python3
"""
Backgammon Video Analysis - Main Entry Point

This tool scrapes backgammon match videos from YouTube, downloads associated
XG (eXtreme Gammon) analysis files, and trains a video model to predict
board states from video frames.
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_scrape(args):
    """Scrape video metadata from YouTube channel."""
    from src.scraper import BackgammonCafeScraper

    scraper = BackgammonCafeScraper(output_dir=args.output_dir)

    print(f"Fetching videos from BackgammonCafe channel...")
    limit = args.limit if args.limit > 0 else None
    videos = scraper.get_channel_videos(limit=limit)

    print(f"\nFound {len(videos)} videos")

    # Show summary
    with_xg = sum(1 for v in videos if v.xg_file_url)
    print(f"Videos with detected XG links: {with_xg}")

    # Save metadata
    scraper.save_metadata(videos, args.output_file)
    print(f"\nMetadata saved to {args.output_dir}/{args.output_file}")


def cmd_download_videos(args):
    """Download videos from YouTube."""
    from src.scraper import BackgammonCafeScraper, VideoDownloader

    scraper = BackgammonCafeScraper(output_dir=args.metadata_dir)
    videos = scraper.load_metadata(args.metadata_file)

    if args.limit > 0:
        videos = videos[:args.limit]

    downloader = VideoDownloader(
        output_dir=args.output_dir,
        quality=args.quality
    )

    print(f"Downloading {len(videos)} videos...")
    results = downloader.download_batch(videos, max_workers=args.workers)

    success = sum(1 for v in results.values() if v is not None)
    print(f"\nDownloaded {success}/{len(videos)} videos")


def cmd_download_xg(args):
    """Download XG files from detected URLs."""
    import json
    from src.xg_parser import XGDownloader

    # Load video metadata
    with open(args.metadata_file, 'r') as f:
        videos = json.load(f)

    # Filter videos with XG URLs
    xg_urls = [v['xg_file_url'] for v in videos if v.get('xg_file_url')]

    if not xg_urls:
        print("No XG file URLs found in metadata.")
        print("Note: XG files may need to be manually obtained from:")
        print("  - https://shop.backgammongalaxy.com/ (UBC match files)")
        print("  - https://thebackgammoncafe.com/ (membership)")
        return

    downloader = XGDownloader(output_dir=args.output_dir)

    print(f"Downloading {len(xg_urls)} XG files...")
    results = downloader.download_batch(xg_urls)

    success = sum(1 for v in results.values() if v is not None)
    print(f"\nDownloaded {success}/{len(xg_urls)} XG files")


def cmd_parse_xg(args):
    """Parse XG files and show summary."""
    from src.xg_parser import XGReader
    import json

    xg_dir = Path(args.xg_dir)
    xg_files = list(xg_dir.glob("*.xg"))

    if not xg_files:
        print(f"No XG files found in {xg_dir}")
        return

    print(f"Parsing {len(xg_files)} XG files...")

    results = []
    for xg_file in xg_files:
        try:
            reader = XGReader(str(xg_file))
            match = reader.read()
            results.append({
                "file": str(xg_file),
                "data": reader.to_dict()
            })
            print(f"  {xg_file.name}: {match.player1} vs {match.player2}, "
                  f"{len(match.games)} games")
        except Exception as e:
            print(f"  {xg_file.name}: Error - {e}")

    # Save parsed data
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nParsed data saved to {args.output_file}")


def cmd_train(args):
    """Train the video analysis model."""
    from src.model import Trainer
    from src.model.trainer import TrainingConfig

    config = TrainingConfig(
        video_dir=args.video_dir,
        xg_dir=args.xg_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone=args.backbone
    )

    trainer = Trainer(config)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    train_loader, val_loader = trainer.create_dataloaders()

    if len(train_loader.dataset) == 0:
        print("No training data found!")
        print("\nTo train the model, you need:")
        print("1. Videos in data/videos/")
        print("2. Matching XG files in data/xg_files/")
        print("3. A manifest mapping videos to XG files")
        return

    print(f"Training on {len(train_loader.dataset)} samples")
    trainer.train(train_loader, val_loader)


def cmd_infer(args):
    """Run inference on a video."""
    import torch
    from src.model import VideoBackgammonModel
    from src.video import FrameExtractor

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoBackgammonModel().to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Extract frames
    extractor = FrameExtractor(resize=(224, 224), normalize=True)
    batch = extractor.extract_frames(args.video, num_frames=16)

    # Prepare input
    frames = torch.from_numpy(batch.frames).permute(3, 0, 1, 2).unsqueeze(0)
    frames = frames.to(device)

    # Run inference
    with torch.no_grad():
        output = model(frames)

    print("Predicted board position:")
    print(output["position"].cpu().numpy())


def main():
    parser = argparse.ArgumentParser(
        description="Backgammon Video Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape video metadata
  python main.py scrape --limit 50

  # Download videos
  python main.py download-videos --limit 10 --quality 720p

  # Parse XG files
  python main.py parse-xg --xg-dir data/xg_files

  # Train model
  python main.py train --epochs 50 --batch-size 8

  # Run inference
  python main.py infer --video path/to/video.mp4 --checkpoint checkpoints/best_model.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape video metadata")
    scrape_parser.add_argument("--output-dir", default="data/videos")
    scrape_parser.add_argument("--output-file", default="video_metadata.json")
    scrape_parser.add_argument("--limit", type=int, default=0)

    # Download videos command
    dl_videos = subparsers.add_parser("download-videos", help="Download videos")
    dl_videos.add_argument("--metadata-dir", default="data/videos")
    dl_videos.add_argument("--metadata-file", default="video_metadata.json")
    dl_videos.add_argument("--output-dir", default="data/videos")
    dl_videos.add_argument("--quality", default="720p")
    dl_videos.add_argument("--limit", type=int, default=0)
    dl_videos.add_argument("--workers", type=int, default=2)

    # Download XG command
    dl_xg = subparsers.add_parser("download-xg", help="Download XG files")
    dl_xg.add_argument("--metadata-file", default="data/videos/video_metadata.json")
    dl_xg.add_argument("--output-dir", default="data/xg_files")

    # Parse XG command
    parse_xg = subparsers.add_parser("parse-xg", help="Parse XG files")
    parse_xg.add_argument("--xg-dir", default="data/xg_files")
    parse_xg.add_argument("--output-file", default="data/xg_files/parsed_data.json")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--video-dir", default="data/videos")
    train_parser.add_argument("--xg-dir", default="data/xg_files")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--backbone", default="resnet18")
    train_parser.add_argument("--resume", default=None)

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--video", required=True)
    infer_parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")

    args = parser.parse_args()

    if args.command == "scrape":
        cmd_scrape(args)
    elif args.command == "download-videos":
        cmd_download_videos(args)
    elif args.command == "download-xg":
        cmd_download_xg(args)
    elif args.command == "parse-xg":
        cmd_parse_xg(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
