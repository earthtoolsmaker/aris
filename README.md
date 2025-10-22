# ARIS Sonar Processing Toolkit

A Python toolkit for processing ARIS (Adaptive Resolution Imaging Sonar) files. Convert sonar data to video, apply motion detection preprocessing, and stabilize jittery footage.

## Demo

https://github.com/user-attachments/assets/b4fd0268-79d6-4215-b934-3c3f7b5169a9

**Left:** Raw ARIS sonar data converted to video
**Right:** Preprocessed output showing detected motion (dual-channel visualization: blue=input, red=motion)

## What is ARIS?

ARIS is a high-frequency imaging sonar system used for underwater imaging and mapping. Unlike traditional optical cameras that rely on light, ARIS uses sound waves to create detailed images in murky water, darkness, or sediment-laden environments. This toolkit processes ARIS sonar recordings into analyzable video formats and applies advanced preprocessing for fish detection and motion analysis.

## Features

- **ARIS to Video Conversion**: Convert proprietary ARIS files to standard MP4 format
- **Video Stabilization**: Reduce temporal jitter using bidirectional Gaussian smoothing
- **Motion Detection**: Preprocessing pipeline with Gaussian blur, MOG2 background subtraction, guided filtering, and temporal smoothing
- **Dual-Channel Visualization**: RGB output showing both input (blue) and detected motion (red) for easy analysis
- **Combined Processing**: Single-pass stabilization + preprocessing for maximum efficiency
- **Batch Processing**: Process entire directories with progress tracking
- **Video Utilities**: Chunking, codec conversion, frame extraction, and averaging

## Setup

### Requirements

- **Python**: 3.13+ (managed with `uv`)
- **FFmpeg**: [Download here](https://www.ffmpeg.org/) - A complete, cross-platform solution to record, convert and stream audio and video

### Python Dependencies

Install `uv` with `pipx`:

```sh
pipx install uv
```

Create a virtualenv and install the dependencies with `uv`:

```sh
uv sync
```

Activate the `uv` virtualenv:

```sh
source .venv/bin/activate
```

## Scripts

### Core Conversion

**Convert ARIS files to MP4 videos:**

Convert all ARIS files from a directory:

```bash
uv run python ./scripts/convert_aris_to_video.py \
  --dir-aris ./data/aris/jansen-lake-2025/ARIS_2025_05_06 \
  --dir-save ./data/mp4/jansen-lake-2025/ARIS_2025_05_06
```

Convert a single ARIS file:

```bash
uv run python ./scripts/convert_aris_to_video.py \
  --filepath-aris ./data/aris/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_000000.aris \
  --dir-save ./data/mp4/jansen-lake-2025/ARIS_2025_05_06
```

### Video Processing Pipeline

These scripts apply advanced preprocessing to sonar videos for motion detection and analysis.

**Combined pipeline (recommended)** - Stabilize and preprocess in a single pass (more efficient):

```bash
uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \
  --filepath-video ./data/mp4/jansen-lake-2025/2025-05-06_001500.mp4 \
  --filepath-save ./data/processed/2025-05-06_001500.mp4
```

The preprocessed output is an RGB video with dual-channel visualization:
- **Blue channel**: Gaussian-blurred input (shows sonar structure)
- **Red channel**: Preprocessed output (shows detected motion)
- **Magenta/purple**: Motion overlapping with sonar structures

**Alternative: Individual scripts** - Run stabilization and preprocessing separately:

Stabilize sonar video using bidirectional Gaussian temporal smoothing:

```bash
uv run python ./scripts/stabilize_sonar_video.py \
  --filepath-video ./data/mp4/jansen-lake-2025/2025-05-06_001500.mp4 \
  --filepath-save ./data/stabilized/2025-05-06_001500.mp4 \
  --window-size 5 \
  --sigma 1.0
```

Preprocess stabilized video for motion detection:

```bash
uv run python ./scripts/preprocess_sonar_video.py \
  --filepath-video ./data/stabilized/2025-05-06_001500.mp4 \
  --filepath-save ./data/preprocessed/2025-05-06_001500.mp4 \
  --gaussian-kernel 3 \
  --mog-history 500 \
  --guided-radius 10
```

### Video Utilities

**Extract average frame** from a video (useful for thumbnails):

```bash
uv run python ./scripts/extract_average_video_frame.py \
  --filepath-video ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_233000.mp4 \
  --filepath-save ./data/jpg_average_frame/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_233000.jpg
```

**Encode videos with new codec** (e.g., H.264 for better compatibility):

```bash
uv run python ./scripts/encode_video_with_codec.py \
  --dir-videos ./data/mp4/ \
  --dir-save ./export/mp4_h264 \
  --video-codec "h264"
```

**Chunk large videos** into non-overlapping segments:

Chunk a single video file:

```bash
uv run python ./scripts/chunk_video.py \
  --filepath-video ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_000000.mp4 \
  --dir-save ./data/chunks/jansen-lake-2025/ARIS_2025_05_06/ \
  --duration-seconds 120
```

Chunk all videos in a directory:

```bash
uv run python ./scripts/chunk_video.py \
  --dir-videos ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/ \
  --dir-save ./data/chunks/jansen-lake-2025/ARIS_2025_05_06/ \
  --duration-seconds 120
```

## Typical Workflow

### Recommended: 3-Step Pipeline

The most efficient workflow for processing ARIS sonar files:

```bash
# Step 1: Convert ARIS files to MP4 videos
uv run python ./scripts/convert_aris_to_video.py \
  --dir-aris ./data/aris/location/ \
  --dir-save ./data/mp4/location/

# Step 2: Stabilize and preprocess for motion detection (single-pass, efficient)
# Output: RGB video with blue=input, red=detected motion
uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \
  --filepath-video ./data/mp4/location/file.mp4 \
  --filepath-save ./data/processed/file.mp4

# Step 3 (Optional): Chunk large videos into manageable segments
uv run python ./scripts/chunk_video.py \
  --filepath-video ./data/processed/file.mp4 \
  --dir-save ./data/chunks/ \
  --duration-seconds 120
```

### Alternative: 4-Step Pipeline

If you need separate control over stabilization and preprocessing:

```bash
# Step 1: Convert ARIS to MP4
uv run python ./scripts/convert_aris_to_video.py \
  --dir-aris ./data/aris/location/ \
  --dir-save ./data/mp4/location/

# Step 2: Stabilize video (reduce temporal jitter)
uv run python ./scripts/stabilize_sonar_video.py \
  --filepath-video ./data/mp4/location/file.mp4 \
  --filepath-save ./data/stabilized/file.mp4

# Step 3: Preprocess for motion detection
uv run python ./scripts/preprocess_sonar_video.py \
  --filepath-video ./data/stabilized/file.mp4 \
  --filepath-save ./data/preprocessed/file.mp4

# Step 4 (Optional): Chunk videos
uv run python ./scripts/chunk_video.py \
  --filepath-video ./data/preprocessed/file.mp4 \
  --dir-save ./data/chunks/ \
  --duration-seconds 120
```
