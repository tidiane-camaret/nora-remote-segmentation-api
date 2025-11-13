# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Nora Remote Segmentation API** is a FastAPI-based REST API server that provides AI-powered 3D medical image segmentation capabilities for the [Nora imaging platform](https://www.nora-imaging.com/). The server enables interactive segmentation refinement through multiple interaction modalities (points, bounding boxes, scribbles, and ROI masks).

## Development Commands

### Setup
```bash
# Install dependencies using uv (fast Python package manager)
uv sync
```

### Running the Server
```bash
# Basic server startup (default: port 1527, RAM-only cache)
uv run python -m main

# With command-line options:
uv run python -m main --host 0.0.0.0 --port 1527 \
    --cache-dir ./cache \
    --log-file \
    --log-level INFO \
    --ssl_keyfile ./certs/key.pem \
    --ssl_certfile ./certs/cert.pem
```

**Command-line arguments:**
- `--host` - Host interface to bind to (default: 0.0.0.0)
- `--port` - Port to listen on (default: 1527)
- `--cache-dir` - Directory for persistent cache storage. Use 'none' to disable persistence (default: none - RAM only)
- `--log-file` - Enable logging to file in `logs/` directory (default: console only)
- `--log-level` - Set logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--ssl_keyfile` - Path to SSL key file for HTTPS
- `--ssl_certfile` - Path to SSL certificate file for HTTPS

The server uses uvicorn with auto-reload enabled, so file changes will automatically restart the server.

### Performance Testing
```bash
# Run predictions on test dataset and compute metrics (DSC, NSD, running time)
uv run python -m tests.test_performance
```

Tests run on images in `config["DATA_DIR"]` (npz format) and evaluate different prompt types (points, bboxes, ROIs).

### Code Formatting
```bash
# Format code with Black (88 char line length)
pre-commit run black --all-files

# Or install pre-commit hooks (runs automatically on git commit)
pre-commit install
```

## Architecture Overview

### Core Request Flow
```
Client (Nora Web UI)
    ↓
FastAPI Endpoints (main.py)
    ↓
Global State Management (PROMPT_MANAGER, IMAGE_CACHE, ROI_CACHE)
    ↓
PromptManager (src/prompt_manager.py) → nnInteractive Model
    ↓
Response (compressed binary ROI data)
```

### State Management (CRITICAL)

**Global State Pattern:** The server uses global variables to manage active sessions:
- `PROMPT_MANAGER` - Singleton inference session manager
- `CURRENT_IMAGE_HASH` - Tracks which image is currently loaded
- `CURRENT_ROI_UUID` - Tracks interaction history for current ROI
- `IMAGE_CACHE` / `ROI_CACHE` - Two-tier (RAM + disk) LRU caches

**⚠️ Known Architectural Issue:** Global state creates race conditions in multi-client scenarios. When multiple clients send concurrent requests, they can interfere with each other because they share the same `PROMPT_MANAGER` and `CURRENT_IMAGE_HASH`. This is documented in `docs/serve_infra_analysis.md` as a CRITICAL issue requiring session-based architecture refactoring.

### Coordinate Systems (CRITICAL)

**Two different coordinate conventions are used:**
- **Client (Nora/JavaScript):** `[x, y, z]` in world/voxel space
- **Server (NumPy arrays):** `[z, y, x]` in array indexing

All API endpoints handle coordinate transformations at the boundary. When working with coordinates, always verify which convention is being used.

### Caching Strategy

**Two-tier cache system:**
- **IMAGE_CACHE:** 5 GB RAM, 30 GB disk, stores full 3D images
- **ROI_CACHE:** 512 MB RAM (compressed), 2 GB disk, stores segmentation masks

**Cache persistence:**
- Controlled by `--cache-dir` argument
- Use `--cache-dir none` for RAM-only (default)
- Use `--cache-dir ./cache` for persistent disk cache
- Cache uses LRU eviction policy
- Persists across server restarts when disk cache enabled

**Cache keys:**
- Image hashes: xxHash of image array data
- ROI hashes: xxHash of compressed ROI binary data
- ⚠️ Client uses weak hash function (documented as CRITICAL issue in `docs/serve_infra_analysis.md`)

### Interaction Session Logic

**Session reset conditions:**
1. ROI UUID changes between requests
2. `roi_has_changed=True` flag in request
3. Active image changes via `/set_active_image`

**Interaction accumulation:**
- Points, bboxes, scribbles accumulate within a session
- Each interaction refines the previous segmentation
- Call `/reset_interactions` to clear accumulated prompts

## Key API Endpoints

**Image Management:**
- `POST /upload_image` - Upload 3D image or load from server-side path
- `GET /check_image/{image_hash}` - Check if image is cached
- `POST /set_active_image/{image_hash}` - Switch active image from cache

**Interactive Segmentation:**
- `POST /add_roi_interaction` - Segment using ROI mask as prompt
- `POST /add_bbox_interaction` - Refine with bounding box
- `POST /add_scribble_interaction` - Refine with scribbles
- `POST /add_point_interaction` - Refine with point(s)

**Session Management:**
- `POST /reset_interactions` - Clear accumulated interactions
- `POST /upload_roi` - Pre-cache ROI without segmentation

## File Organization

**Entry points:**
- `main.py` - Primary FastAPI server (816 lines)
- `main_3dslicer.py` - Legacy/alternative server for 3D Slicer integration

**Core library (`src/`):**
- `src/prompt_manager.py` - Wraps nnInteractive inference session, manages model loading from HuggingFace
- `src/utils.py` - ArrayCache implementation, memory monitoring (Slurm/cgroup aware), compression, logging
- `src/mock_session.py` - Mock inference session for testing without GPU

**Configuration:**
- `config.yaml` - Machine-specific data paths for performance testing (nora/meta clusters)
- `pyproject.toml` - Dependencies managed with uv

**Ignored directories:**
- `.nninteractive_weights/` - Model weights cache (downloaded from HuggingFace on first run)
- `certs/` - SSL certificates
- `logs/` - Server logs (when `--log-file` enabled)

## Technology Stack

**Deep Learning:**
- PyTorch 2.7.1+ with CUDA support (optional, has mock mode)
- `nninteractive` 1.0.1+ - Interactive segmentation library
- Model weights auto-downloaded from HuggingFace Hub on first run

**Data Processing:**
- NumPy for array operations
- `nibabel` for medical imaging formats (NIfTI)
- `surface-distance` for segmentation metrics (DSC, NSD) - local dependency from `../repos/surface-distance`

**Binary Serialization:**
- Gzip compression for ROI data transfer
- Bit packing for boolean segmentation masks (8x compression)
- xxHash for fast cache key generation

**Memory Management:**
- Slurm/cgroup-aware memory monitoring
- `psutil` for system memory tracking
- `pynvml` for GPU monitoring

## Important Implementation Details

### Dependency Injection Pattern
The `get_active_image()` function in `main.py` is used as a FastAPI dependency to load the active image on-demand. It checks if `CURRENT_IMAGE_HASH` matches the loaded image in `PROMPT_MANAGER`.

### Mock Mode for Testing
`PromptManager` supports `mock=True` for testing without GPU. Mock mode returns a dilated version of the input ROI instead of running the model.

### Memory Monitoring
The server logs memory usage and is Slurm-aware. Use `src.utils.get_slurm_memory_limit()` and `get_job_cgroup_memory_usage()` to query available resources.

### Model Weight Management
First run downloads ~1GB of model weights from HuggingFace to `.nninteractive_weights/`. Subsequent runs load from local cache.

## Known Issues & Architecture Flaws

See `docs/serve_infra_analysis.md` for comprehensive analysis. Critical issues include:

1. **Hash collision vulnerability** - Client uses weak hash function for image IDs
2. **Race conditions in global state** - Multi-client deployments broken due to shared state
3. **No authentication/authorization** - CORS allows all origins (marked TODO)

When working on features related to session management or multi-client support, review the architectural analysis document first.

## Configuration for Performance Testing

The `config.yaml` file contains machine-specific paths for the test dataset:
- `nora` - Paths on nora cluster
- `meta` - Paths on meta cluster

Tests load from `config["DATA_DIR"]` based on the detected machine (see `tests/config.py`).
