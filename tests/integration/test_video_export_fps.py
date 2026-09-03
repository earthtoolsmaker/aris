"""Regression: VideoExport must accept ARIS-like fractional fps under mpeg4."""

import os
from pathlib import Path

import av
import pytest

from aris.pyARIS import pyARIS

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

_DEFAULT_ARIS_DIR = Path.home() / "Projects" / "data" / "arisfiles"

_FIXTURES_ARIS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "aris"


def _is_lfs_pointer(path: Path) -> bool:
    """True if path is a Git LFS pointer stub (not the real object)."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            first = f.readline()
    except OSError:
        return False
    return first.startswith(_LFS_POINTER_PREFIX)


def _usable_aris(path: Path) -> bool:
    """True if path exists and is not a Git LFS pointer stub."""
    return path.is_file() and not _is_lfs_pointer(path)


def _aris_dirs() -> list[Path]:
    """Candidate directories: ARIS_TEST_DIR, fixtures, then ~/Projects default."""
    dirs: list[Path] = []
    env = os.environ.get("ARIS_TEST_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    env_file = os.environ.get("ARIS_TEST_FILE")
    if env_file:
        parent = Path(env_file).expanduser().resolve().parent
        if parent not in dirs:
            dirs.append(parent)
    dirs.append(_FIXTURES_ARIS_DIR)
    dirs.append(_DEFAULT_ARIS_DIR)
    return dirs


def _collect_aris_files() -> list[Path]:
    """Usable .aris files from the first directory that has any."""
    for directory in _aris_dirs():
        if not directory.is_dir():
            continue
        batch = sorted(p for p in directory.glob("*.aris") if _usable_aris(p))
        if batch:
            return batch
    return []


def _aris_skip_reason() -> str:
    fixture = _FIXTURES_ARIS_DIR / "2025-05-13_092300.aris"
    if _is_lfs_pointer(fixture):
        return (
            f"{fixture} is a Git LFS pointer, not the real file. "
            "Run `git lfs pull`, set ARIS_TEST_DIR / ARIS_TEST_FILE, "
            f"or place .aris samples in {_DEFAULT_ARIS_DIR}"
        )
    return (
        "No usable ARIS files: need Git LFS fixtures, ARIS_TEST_DIR / "
        f"ARIS_TEST_FILE, or .aris samples in {_DEFAULT_ARIS_DIR}"
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "aris_path" not in metafunc.fixturenames:
        return
    files = _collect_aris_files()
    if not files:
        metafunc.parametrize(
            "aris_path",
            [pytest.param(None, id="no-aris-files")],
        )
        return
    metafunc.parametrize(
        "aris_path",
        [pytest.param(p, id=p.name) for p in files],
    )


@pytest.mark.slow
def test_video_export_source_fps_writes_two_frames(
    aris_path: Path | None, tmp_path: Path
):
    """Export two frames at each file's native fps; must complete and write."""
    if aris_path is None:
        pytest.skip(_aris_skip_reason())

    data, first = pyARIS.DataImport(str(aris_path))
    fps = float(first.framerate) if first.framerate > 0 else 24.0
    out = tmp_path / f"{aris_path.stem}_two_frames.mp4"

    pyARIS.VideoExport(
        data,
        str(out),
        fps=fps,
        start_frame=1,
        end_frame=2,
        timestamp=False,
        show_progress=False,
    )

    assert out.exists()
    assert out.stat().st_size > 0

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    assert frames == 2
