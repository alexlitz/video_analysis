# Backgammon Video Analysis

A machine learning system for analyzing backgammon match videos and predicting game states using XG (eXtreme Gammon) file format data as ground truth.

## Overview

This project provides tools to:

1. **Scrape** video metadata from the [BackgammonCafe YouTube channel](https://www.youtube.com/@BackgammonCafe)
2. **Download** match videos and associated XG analysis files
3. **Parse** XG files to extract board positions, moves, and analysis
4. **Train** a video model to predict board states from video frames

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape video metadata
python main.py scrape --limit 50

# Download videos (requires yt-dlp)
python main.py download-videos --limit 10 --quality 720p

# Parse XG files
python main.py parse-xg --xg-dir data/xg_files

# Train model
python main.py train --epochs 50 --batch-size 8
```

## Project Structure

```
video_analysis/
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── configs/
│   └── default.yaml        # Default configuration
├── src/
│   ├── scraper/            # YouTube scraping
│   │   ├── youtube_scraper.py
│   │   └── video_downloader.py
│   ├── xg_parser/          # XG file parsing
│   │   ├── xg_reader.py
│   │   └── xg_downloader.py
│   ├── video/              # Video processing
│   │   ├── processor.py
│   │   └── frame_extractor.py
│   └── model/              # ML model
│       ├── dataset.py
│       ├── video_model.py
│       └── trainer.py
├── data/
│   ├── videos/             # Downloaded videos
│   ├── xg_files/           # XG analysis files
│   └── processed/          # Processed data
└── checkpoints/            # Model checkpoints
```

## XG File Format

XG files are binary files from [eXtreme Gammon](https://www.extremegammon.com/), the premier backgammon analysis software. They contain:

- **Match metadata**: Player names, event, date, match length
- **Game records**: Board positions, moves, cube actions
- **Analysis data**: Move evaluations, equity calculations, rollout results

### File Structure

```
[Header: 8232 bytes]
├── Magic number: "HMGR"
├── Version info
├── Thumbnail (JPG)
├── Game GUID
└── Unicode strings (name, comments)

[Compressed Archive]
├── temp.xg    - Game records (2560 bytes each)
├── temp.xgi   - Index file
├── temp.xgr   - Rollout analysis (2184 bytes each)
└── temp.xgc   - Comments (RTF format)
```

### Record Types

| Type | Name | Description |
|------|------|-------------|
| 0 | HeaderMatch | Match metadata (players, ELO, settings) |
| 1 | HeaderGame | Game initialization |
| 2 | Cube | Doubling cube actions |
| 3 | Move | Checker plays with analysis |
| 4 | FooterGame | Game results |
| 5 | FooterMatch | Match conclusion |

### Position Encoding

Board positions use 26 values representing checker counts:
- Points 0-23: The 24 points (negative = opponent's checkers)
- Point 24: Bar
- Point 25: Borne off

### Obtaining XG Files

XG files for backgammon matches can be obtained from:
- [Backgammon Galaxy Shop](https://shop.backgammongalaxy.com/) - UBC tournament matches (free with donation option)
- [The Backgammon Cafe](https://www.thebackgammoncafe.com/) - Membership subscription
- Creating your own using eXtreme Gammon software

---

## Comparison: Ping Pong/Table Tennis Match Data Formats

For comparison, here's how similar video analysis projects work for table tennis:

### Available Datasets

#### 1. OpenTTGames Dataset
- **Source**: [lab.osai.ai](https://lab.osai.ai/)
- **Format**: Full-HD videos at 120 fps + JSON annotations
- **Content**:
  - Ball coordinates (x, y per frame)
  - In-game events (bounces, net hits)
  - Segmentation masks (humans, table, scoreboard)
- **Size**: 5 training videos (10-25 min each) + 7 test videos

#### 2. SPIN Dataset
- **Source**: [arxiv.org/abs/1912.06640](https://arxiv.org/abs/1912.06640)
- **Format**: High-resolution stereo video
- **Content**:
  - Ball tracking (3D position)
  - Human poses
  - Ball spin data
- **Size**: 53 hours training, 1 hour test

#### 3. TTSwing Dataset
- **Source**: [Nature Scientific Data](https://www.nature.com/articles/s41597-025-04680-y)
- **Format**: 9-axis sensor data from racket grips
- **Content**:
  - Acceleration (X, Y, Z)
  - Gyroscope data
  - Player demographics
- **Use case**: Swing analysis, not video-based

#### 4. P2ANet Dataset
- **Source**: [arxiv.org/abs/2207.12730](https://arxiv.org/abs/2207.12730)
- **Format**: Broadcasting video clips
- **Content**:
  - 2,721 video clips from professional matches
  - 14 action classes
  - Dense temporal annotations
- **Use case**: Action detection and recognition

### Comparison Table

| Aspect | Backgammon (XG) | Table Tennis |
|--------|-----------------|--------------|
| **Data Source** | XG software export | Video + sensors |
| **Ground Truth** | Discrete game states | Continuous trajectories |
| **Frame Rate Needed** | Low (1-5 fps) | High (60-120 fps) |
| **Primary Challenge** | Board/checker detection | Fast ball tracking |
| **State Complexity** | 26 checker positions + cube | 3D ball position + spin |
| **Action Types** | Moves, cube decisions | Serves, returns, spins |
| **Existing ML Work** | Limited | Multiple datasets/models |

### Key Differences

1. **Temporal Resolution**
   - Backgammon: State changes are discrete (one move at a time)
   - Ping Pong: Requires high-speed capture for ball tracking

2. **Computer Vision Challenges**
   - Backgammon: Static board, clear checker positions
   - Ping Pong: Fast-moving small ball, motion blur

3. **Ground Truth Availability**
   - Backgammon: XG files provide complete game state
   - Ping Pong: Requires manual annotation or sensor data

4. **Model Architecture**
   - Backgammon: Can use standard image classification + temporal modeling
   - Ping Pong: Requires specialized tracking models (optical flow, 3D CNNs)

### No Universal Ping Pong Match Format

Unlike backgammon's XG format, there is **no standardized file format** for ping pong match data. Each dataset uses its own format:

```python
# OpenTTGames format (JSON)
{
  "frame_num": 1234,
  "ball": {"x": 320, "y": 240, "visible": true},
  "event": "bounce"
}

# SPIN format (custom binary + JSON)
{
  "timestamp": 0.0123,
  "ball_3d": [1.2, 0.8, 0.3],
  "spin": [500, 200, 100]  # rpm
}

# P2ANet format (annotations)
{
  "video_id": "match_001",
  "actions": [
    {"start": 1.2, "end": 2.1, "label": "forehand_topspin"}
  ]
}
```

### Implications for Video Analysis

For training video models:

| Task | Backgammon Advantage | Ping Pong Advantage |
|------|---------------------|---------------------|
| Data availability | Limited sources | Multiple public datasets |
| Annotation effort | Low (XG files exist) | High (manual labeling) |
| Model complexity | Simpler (static scenes) | More complex (motion) |
| Real-time inference | Feasible | Requires optimization |

---

## Model Architecture

The video model uses a CNN backbone + Transformer for temporal modeling:

```
Video Frames (B, C, T, H, W)
         │
         ▼
┌─────────────────┐
│  Frame Encoder  │  (ResNet-18/50)
│  (per frame)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Positional    │
│   Encoding      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer    │  (4 layers, 8 heads)
│  Encoder        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │
│  Heads          │
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
Position   Dice    Cube
(28-dim)  (2×6)   (7-class)
```

## Training

```bash
# Basic training
python main.py train --epochs 100

# With custom settings
python main.py train \
  --video-dir data/videos \
  --xg-dir data/xg_files \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.0001 \
  --backbone resnet50

# Resume training
python main.py train --resume checkpoints/checkpoint_epoch_50.pt
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- yt-dlp (for video downloading)
- OpenCV (for video processing)

See `requirements.txt` for full dependencies.

## License

This project is for educational and research purposes.

- XG file format documentation: [extremegammon.com/xgformat.aspx](https://www.extremegammon.com/xgformat.aspx)
- XG parsing code based on [xgdatatools](https://github.com/oysteijo/xgdatatools) (LGPL-2.1)

## References

### Backgammon
- [eXtreme Gammon](https://www.extremegammon.com/) - XG software
- [Backgammon Galaxy](https://shop.backgammongalaxy.com/) - XG files for tournaments
- [The Backgammon Cafe](https://www.thebackgammoncafe.com/) - Video content

### Table Tennis Datasets
- [OpenTTGames](https://lab.osai.ai/) - Ball detection and event spotting
- [SPIN Dataset](https://arxiv.org/abs/1912.06640) - High-speed tracking
- [TTSwing](https://www.nature.com/articles/s41597-025-04680-y) - Sensor-based analysis
- [P2ANet](https://arxiv.org/abs/2207.12730) - Action detection benchmark
