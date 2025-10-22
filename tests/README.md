# ARIS Test Suite

This directory contains the pytest test suite for the ARIS sonar file processing toolkit.

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures for all tests
├── unit/                 # Unit tests for individual modules
│   ├── test_frame.py     # Tests for frame manipulation utilities
│   └── test_video_utils.py  # Tests for video processing utilities
├── integration/          # Integration tests (future)
└── fixtures/             # Binary test data files (Git LFS)
    ├── aris/             # Sample ARIS sonar files
    └── mp4/              # Sample MP4 video files
```

## Git LFS Setup (Required)

This test suite uses **Git LFS (Large File Storage)** for binary test fixtures (ARIS and MP4 files in `tests/fixtures/`).

### First-Time Setup

If you haven't used Git LFS before:

```bash
# Install Git LFS (one-time system setup)
brew install git-lfs              # macOS
# sudo apt-get install git-lfs    # Ubuntu/Debian
# sudo pacman -S git-lfs          # Arch Linux

# Initialize LFS in your local repository
git lfs install

# Pull the LFS files
git lfs pull
```

### Verify LFS Files

Check that fixtures were downloaded correctly:

```bash
ls -lh tests/fixtures/aris/
ls -lh tests/fixtures/mp4/
```

Files should show their actual sizes (~60-65MB each), not small pointer files.

### For Contributors

When cloning the repository:

```bash
git clone <repository-url>
cd aris
git lfs install
git lfs pull
```

**Note:** If you forget `git lfs pull`, the fixture files will be small text pointers instead of actual binary files, and tests using real ARIS/MP4 files will fail.

## Test Coverage

### What's Tested
- **frame.py**: Frame manipulation utilities
  - Grayscale to RGB conversion
  - Frame type detection (grayscale vs RGB)
  - UTC timestamp extraction from ARIS frames
  - Video generation from frame sequences

- **video/utils.py**: Video processing operations
  - Frame extraction from videos
  - Average frame calculation
  - Video metadata reading (FPS, duration)
  - Video encoding and saving

### What's NOT Tested
- **pyARIS/**: ARIS file parser (excluded - requires binary test files and is third-party code)

## Running Tests

### Install Test Dependencies

First, sync the development dependencies:

```bash
uv sync --dev
```

### Run All Tests

```bash
uv run pytest
```

### Run Tests with Coverage Report

```bash
uv run pytest --cov=src/aris --cov-report=term-missing
```

### Run Specific Test File

```bash
uv run pytest tests/unit/test_frame.py
```

### Run Specific Test Class

```bash
uv run pytest tests/unit/test_frame.py::TestGrayscaleToRgb
```

### Run Specific Test Function

```bash
uv run pytest tests/unit/test_frame.py::TestGrayscaleToRgb::test_converts_grayscale_to_rgb_shape
```

### Run Tests with Verbose Output

```bash
uv run pytest -v
```

### Run Tests and Stop on First Failure

```bash
uv run pytest -x
```

### Run Tests Matching a Pattern

```bash
uv run pytest -k "grayscale"
```

## Test Configuration

Test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "-v",                          # Verbose output
    "--cov=src/aris",              # Coverage for src/aris
    "--cov-report=term-missing",   # Show missing lines
    "--cov-report=html",           # Generate HTML coverage report
    "--ignore=src/aris/pyARIS/",   # Exclude pyARIS from coverage
]
```

## Coverage Reports

After running tests with coverage, you can view:

1. **Terminal output**: Shows coverage percentage and missing lines
2. **HTML report**: Open `htmlcov/index.html` in a browser for detailed coverage

```bash
# Generate and view HTML coverage report
uv run pytest --cov=src/aris --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Writing New Tests

### Using Fixtures

Shared fixtures are defined in `conftest.py`. Common fixtures include:

- `sample_grayscale_frame`: 100x100 grayscale NumPy array
- `sample_rgb_frame`: 100x100 RGB NumPy array
- `sample_video_file`: Temporary MP4 video with 10 frames at 24 FPS
- `mock_aris_frame`: Mock ARIS frame object with timestamp

Example usage:

```python
def test_my_function(sample_grayscale_frame):
    """Test using a fixture."""
    result = my_function(sample_grayscale_frame)
    assert result.shape == (100, 100, 3)
```

### Test Organization

- Group related tests in classes (e.g., `TestGrayscaleToRgb`)
- Use descriptive test names that explain what is being tested
- Follow the pattern: `test_<what>_<condition>_<expected>`
- Add docstrings to explain complex test scenarios

### Example Test Structure

```python
class TestMyFunction:
    """Tests for my_function()."""

    def test_basic_functionality(self):
        """Test that function works with normal input."""
        result = my_function(valid_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test that function handles edge case correctly."""
        result = my_function(edge_case_input)
        assert result is not None

    def test_error_handling(self):
        """Test that function raises appropriate error."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    uv sync --dev
    uv run pytest --cov=src/aris --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you've installed the package in development mode:

```bash
uv sync --dev
```

### FFmpeg Not Found

Some tests require FFmpeg for video encoding. Install it:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg
```

### Video Codec Issues

If video encoding tests fail, it may be due to codec availability. The tests use:
- `mp4v` codec (default, widely supported)
- `libx264` codec (H.264, requires FFmpeg with x264 support)

### Slow Tests

If tests are slow, run only unit tests (skip integration tests):

```bash
uv run pytest tests/unit/
```

## Future Improvements

- [ ] Add integration tests for full workflows
- [ ] Add tests for preprocessing scripts
- [ ] Create test fixtures with small sample ARIS files
- [ ] Add performance/benchmark tests
- [ ] Add property-based tests with Hypothesis
