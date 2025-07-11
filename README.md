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

Convert an ARIS file into an MP4 video:

```bash
uv run python ./scripts/convert_aris_to_video.py \
--filepath-aris your_aris_file.aris \
--dir-save export/ \
--video-codec h264 \
--video-fps 10 \
--loglevel info
```
