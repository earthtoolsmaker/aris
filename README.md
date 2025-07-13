# python aris

Process ARIS files with Python.

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
