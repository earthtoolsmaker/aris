# python sonar aris

Process ARIS sonar files with Python.

## Setup

### Binaries dependencies

- [ffmpeg](https://www.ffmpeg.org/): A complete, cross-platform solution to
record, convert and stream audio and video.

### 🐍 Python dependencies

Install `uv` with `pipx`:

```sh
pipx install uv
```

Create a virtualenv and install the dependencies with `uv`:

```sh
uv sync
```

Activate the `uv` virutalenv:

```sh
source .venv/bin/activate
```

## Scripts

Convert all ARIS files from a directory into MP4 videos:

```bash
uv run python ./scripts/convert_aris_to_video.py \
  --dir-aris ./data/aris/jansen-lake-2025/ARIS_2025_05_06 \
  --dir-save ./data/mp4/jansen-lake-2025/ARIS_2025_05_06
```

Convert one ARIS file into an MP4 video:

```bash
uv run python ./scripts/convert_aris_to_video.py \
  --filepath-aris ./data/aris/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_000000.aris \
  --dir-save ./data/mp4/jansen-lake-2025/ARIS_2025_05_06
```

Extract the average frame from a video:

```bash
uv run python ./scripts/extract_average_video_frame.py \
--filepath-video ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_233000.mp4 \
--filepath-save ./data/jpg_average_frame/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_233000.jpg
```

Encode all mp4 video files with a new codec:

```bash
uv run python ./scripts/encode_video_with_codec.py \
  --dir-videos ./data/mp4/ \
  --dir-save ./export/mp4_h264 \
  --video-codec "h264"
```

Chunk a large video file into non overlapping segments:

```bash
uv run python ./scripts/chunk_video.py \
  --filepath-video ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/2025-05-06_000000.mp4 \
  --dir-save ./data/chunks/jansen-lake-2025/ARIS_2025_05_06/ \
  --duration-seconds 120
```

Chunk a directory of video files into non overlapping segments:

```bash
uv run python ./scripts/chunk_video.py \
  --dir-videos ./data/mp4/jansen-lake-2025/ARIS_2025_05_06/ \
  --dir-save ./data/chunks/jansen-lake-2025/ARIS_2025_05_06/ \
  --duration-seconds 120
```

### Video Processing Pipeline

Stabilize sonar video using bidirectional Gaussian temporal smoothing:

```bash
uv run python ./scripts/stabilize_sonar_video.py \
  --filepath-video ./data/mp4/jansen-lake-2025/2025-05-06_001500.mp4 \
  --filepath-save ./data/stabilized/2025-05-06_001500.mp4 \
  --window-size 5 \
  --sigma 1.0
```

Preprocess sonar video for motion detection (Gaussian blur, MOG2, guided filter, temporal smoothing):

```bash
uv run python ./scripts/preprocess_sonar_video.py \
  --filepath-video ./data/stabilized/2025-05-06_001500.mp4 \
  --filepath-save ./data/preprocessed/2025-05-06_001500.mp4 \
  --gaussian-kernel 3 \
  --mog-history 500 \
  --guided-radius 10
```

**Recommended:** Combine stabilization and preprocessing in a single pass (more efficient):

```bash
uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \
  --filepath-video ./data/mp4/jansen-lake-2025/2025-05-06_001500.mp4 \
  --filepath-save ./data/processed/2025-05-06_001500.mp4
```

The preprocessed output is an RGB video with dual-channel visualization:
- **Blue channel**: Gaussian-blurred input (shows sonar structure)
- **Red channel**: Preprocessed output (shows detected motion)
- **Magenta/purple**: Motion overlapping with sonar structures

## Typical Workflow

Complete pipeline for processing ARIS sonar files:

```bash
# 1. Convert ARIS files to MP4 videos
uv run python ./scripts/convert_aris_to_video.py \
  --dir-aris ./data/aris/location/ \
  --dir-save ./data/mp4/location/

# 2. Stabilize and preprocess for motion detection (combined, recommended)
uv run python ./scripts/stabilize_and_preprocess_sonar_video.py \
  --filepath-video ./data/mp4/location/file.mp4 \
  --filepath-save ./data/processed/file.mp4

# 3. Optional: Chunk processed videos into segments
uv run python ./scripts/chunk_video.py \
  --filepath-video ./data/processed/file.mp4 \
  --dir-save ./data/chunks/ \
  --duration-seconds 120
```

Alternative workflow (separate stabilization and preprocessing):

```bash
# 1. Convert ARIS to MP4
uv run python ./scripts/convert_aris_to_video.py \
  --dir-aris ./data/aris/location/ \
  --dir-save ./data/mp4/location/

# 2a. Stabilize video
uv run python ./scripts/stabilize_sonar_video.py \
  --filepath-video ./data/mp4/location/file.mp4 \
  --filepath-save ./data/stabilized/file.mp4

# 2b. Preprocess stabilized video
uv run python ./scripts/preprocess_sonar_video.py \
  --filepath-video ./data/stabilized/file.mp4 \
  --filepath-save ./data/preprocessed/file.mp4
```
