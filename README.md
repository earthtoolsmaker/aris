# ARIS Sonar Processing Toolkit

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/earthtoolsmaker/aris)
[![CI](https://github.com/earthtoolsmaker/aris/actions/workflows/ci.yml/badge.svg)](https://github.com/earthtoolsmaker/aris/actions/workflows/ci.yml)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/earthtoolsmaker/aris)

A Python toolkit for processing ARIS (Adaptive Resolution Imaging Sonar) files. Convert sonar data to video, apply motion detection preprocessing, and stabilize jittery footage.

> **⚠️ Development Status**: This library is under active development. The API is not yet stable and breaking changes are expected in future releases. Use with caution in production environments and pin your version dependencies.

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

## CLI Commands

After installation, the following commands are available from anywhere in your terminal:

### Core Conversion

**Convert ARIS files to MP4 videos:**

Convert all ARIS files from a directory:

```bash
aris-convert \
  --dir-aris ./data/aris/jansen-lake-2025/ARIS_2025_05_06 \
  --dir-save ./data/mp4/jansen-lake-2025/ARIS_2025_05_06
```

Convert a single ARIS file:

```bash
aris-convert \
  --filepath-aris ./data/aris/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_000000.aris \
  --dir-save ./data/mp4/jansen-lake-2025/ARIS_2025_05_06
```

### Video Processing Pipeline

These commands apply advanced preprocessing to sonar videos for motion detection and analysis.

**Combined pipeline (recommended)** - Stabilize and preprocess in a single pass (more efficient):

```bash
aris-stabilize-preprocess \
  --filepath-video ./data/mp4/jansen-lake-2025/2025-05-06_001500.mp4 \
  --filepath-save ./data/processed/2025-05-06_001500.mp4
```

The preprocessed output is an RGB video with dual-channel visualization:
- **Blue channel**: Gaussian-blurred input (shows sonar structure)
- **Red channel**: Preprocessed output (shows detected motion)
- **Magenta/purple**: Motion overlapping with sonar structures

**Alternative: Individual commands** - Run stabilization and preprocessing separately:

Stabilize sonar video using bidirectional Gaussian temporal smoothing:

```bash
aris-stabilize \
  --filepath-video ./data/mp4/jansen-lake-2025/2025-05-06_001500.mp4 \
  --filepath-save ./data/stabilized/2025-05-06_001500.mp4 \
  --window-size 5 \
  --sigma 1.0
```

Preprocess stabilized video for motion detection:

```bash
aris-preprocess \
  --filepath-video ./data/stabilized/2025-05-06_001500.mp4 \
  --filepath-save ./data/preprocessed/2025-05-06_001500.mp4 \
  --gaussian-kernel 3 \
  --mog-history 500 \
  --guided-radius 10
```

### Video Utilities

**Extract average frame** from a video (useful for thumbnails):

```bash
aris-extract-frame \
  --filepath-video ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_233000.mp4 \
  --filepath-save ./data/jpg_average_frame/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_233000.jpg
```

**Encode videos with new codec** (e.g., H.264 for better compatibility):

```bash
aris-encode \
  --dir-videos ./data/mp4/ \
  --dir-save ./export/mp4_h264 \
  --video-codec "h264"
```

**Chunk large videos** into non-overlapping segments:

Chunk a single video file:

```bash
aris-chunk \
  --filepath-video ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_000000.mp4 \
  --dir-save ./data/chunks/jansen-lake-2025/ARIS_2025_05_06/ \
  --duration-seconds 120
```

Chunk all videos in a directory:

```bash
aris-chunk \
  --dir-videos ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/ \
  --dir-save ./data/chunks/jansen-lake-2025/ARIS_2025_05_06/ \
  --duration-seconds 120
```

## Typical Workflow

### Recommended: 3-Step Pipeline

The most efficient workflow for processing ARIS sonar files:

```bash
# Step 1: Convert ARIS files to MP4 videos
aris-convert \
  --dir-aris ./data/aris/location/ \
  --dir-save ./data/mp4/location/

# Step 2: Stabilize and preprocess for motion detection (single-pass, efficient)
# Output: RGB video with blue=input, red=detected motion
aris-stabilize-preprocess \
  --filepath-video ./data/mp4/location/file.mp4 \
  --filepath-save ./data/processed/file.mp4

# Step 3 (Optional): Chunk large videos into manageable segments
aris-chunk \
  --filepath-video ./data/processed/file.mp4 \
  --dir-save ./data/chunks/ \
  --duration-seconds 120
```

### Alternative: 4-Step Pipeline

If you need separate control over stabilization and preprocessing:

```bash
# Step 1: Convert ARIS to MP4
aris-convert \
  --dir-aris ./data/aris/location/ \
  --dir-save ./data/mp4/location/

# Step 2: Stabilize video (reduce temporal jitter)
aris-stabilize \
  --filepath-video ./data/mp4/location/file.mp4 \
  --filepath-save ./data/stabilized/file.mp4

# Step 3: Preprocess for motion detection
aris-preprocess \
  --filepath-video ./data/stabilized/file.mp4 \
  --filepath-save ./data/preprocessed/file.mp4

# Step 4 (Optional): Chunk videos
aris-chunk \
  --filepath-video ./data/preprocessed/file.mp4 \
  --dir-save ./data/chunks/ \
  --duration-seconds 120
```
