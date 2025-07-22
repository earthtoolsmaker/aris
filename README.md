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

Chunk the mp4 video files into smaller video clips:

```bash
uv run python ./scripts/chunk_video.py \
  --dir-videos ./data/mp4/jansen-lake-2025/ARIS_2025_05_11/ \
  --dir-save ./chunks/jansen-lake-2025/ARIS_2025_05_11/ \
  --duration-seconds 120
```
