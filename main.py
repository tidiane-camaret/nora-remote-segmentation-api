import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.prompt_manager import PromptManager
from src.utils import (
    GPU_AVAILABLE,
    ArrayCache,
    deserialize_array,
    get_job_cgroup_memory_usage,
    get_slurm_memory_limit,
    log_memory_usage,
    serialize_array,
    setup_logging,
)

# Get logger (will be configured in main() or via environment variables)
logger = logging.getLogger(__name__)

# Configure logging at module import if not already configured
# This handles the case where uvicorn worker imports the module
if not logging.getLogger().handlers:
    log_dir = os.environ.get("SEGMENTATION_API_LOG_DIR", "")
    log_level = os.environ.get("SEGMENTATION_API_LOG_LEVEL", "INFO")
    setup_logging(log_dir=log_dir if log_dir else None, log_level=log_level)

# --- Globals & App Initialization ---
# Caches will be initialized in main() or at module import based on environment
IMAGE_CACHE = None
ROI_CACHE = None
PROMPT_MANAGER = PromptManager()
CURRENT_IMAGE_HASH = None  # Tracks the image hash currently loaded in PROMPT_MANAGER
CURRENT_ROI_UUID = None  # Tracks the ROI UUID currently active in PROMPT_MANAGER
CURRENT_CLIM0 = None  # Tracks the lower histogram boundary currently applied
CURRENT_CLIM1 = None  # Tracks the upper histogram boundary currently applied


def init_caches(persist_dir: str | None = None):
    """
    Initialize image and ROI caches with optional persistence.

    Args:
        persist_dir: Directory for persistent cache storage. If None, caches are RAM-only.
    """
    global IMAGE_CACHE, ROI_CACHE

    IMAGE_CACHE = ArrayCache(
        max_size_bytes=5 * 1024 * 1024 * 1024,
        cache_name="Image",
        compress=False,
        persist_dir=persist_dir,
        max_disk_size_bytes=(
            30 * 1024 * 1024 * 1024 if persist_dir else None
        ),  # 30 GB disk storage
    )  # 5 GB RAM, 30 GB disk (if persistence enabled)

    ROI_CACHE = ArrayCache(
        max_size_bytes=512 * 1024 * 1024,
        cache_name="ROI",
        compress=True,
        persist_dir=persist_dir,
        max_disk_size_bytes=(
            2 * 1024 * 1024 * 1024 if persist_dir else None
        ),  # 2 GB disk storage
    )  # 512 MB RAM (compressed), 2 GB disk (if persistence enabled)


# Initialize caches at module import if not already initialized
# This handles the case where uvicorn worker imports the module
if IMAGE_CACHE is None:
    cache_dir_env = os.environ.get("SEGMENTATION_API_CACHE_DIR", "none")
    cache_dir = None if cache_dir_env.lower() == "none" else cache_dir_env
    init_caches(persist_dir=cache_dir)

app = FastAPI()

# Add CORS middleware # TODO : restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Utility Functions ---
def apply_histogram_trimming(image_array: np.ndarray, clim0: float, clim1: float) -> np.ndarray:
    """
    Apply histogram trimming to an image array by clipping values to [clim0, clim1].

    Args:
        image_array: Input image array
        clim0: Lower histogram boundary
        clim1: Upper histogram boundary

    Returns:
        Trimmed image array (clipped to [clim0, clim1])
    """
    logger.info(f"Applying histogram trimming: clim0={clim0}, clim1={clim1}")
    logger.info(f"Image stats before trimming: min={image_array.min():.2f}, max={image_array.max():.2f}, mean={image_array.mean():.2f}")

    # Clip the image values to the histogram boundaries
    trimmed_array = np.clip(image_array, clim0, clim1)

    logger.info(f"Image stats after trimming: min={trimmed_array.min():.2f}, max={trimmed_array.max():.2f}, mean={trimmed_array.mean():.2f}")

    return trimmed_array


def parse_filepath_with_timestamp(path: str) -> Path:
    """
    Extract file path from timestamp-prefixed string.

    Some clients send paths with a timestamp prefix (e.g., "1234567890 /path/to/file.nii").
    This function strips the prefix and returns the actual path.

    Args:
        path: Path string, possibly with timestamp prefix

    Returns:
        Path object for the extracted file path
    """
    path = path.strip()
    if " " in path:
        # Split and take the last part (the actual file path)
        path = path.split(maxsplit=1)[1]
    return Path(path)


async def load_roi_from_cache_or_upload(
    roi_hash: str,
    file: UploadFile | None,
    shape: str,
    dtype: str,
    compressed: str | None,
    interaction_type: str = "interaction"
) -> np.ndarray:
    """
    Load ROI from cache or uploaded file.

    Args:
        roi_hash: Hash identifier for the ROI
        file: Uploaded file containing ROI data (optional if in cache)
        shape: JSON string representing ROI shape
        dtype: Data type of the ROI array
        compressed: Compression type ("gzip" or None)
        interaction_type: Type of interaction (for logging purposes)

    Returns:
        ROI array as numpy array

    Raises:
        HTTPException: If ROI not in cache and no file provided
    """
    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache for {interaction_type}.")
        return cached_roi

    # ROI not cached, need to load from file
    if file is None:
        raise HTTPException(
            status_code=400, detail="ROI not in cache and no file provided"
        )

    binary_data = await file.read()
    shape_tuple = tuple(json.loads(shape))
    is_compressed = compressed == "gzip"
    roi_array = deserialize_array(
        binary_data, shape_tuple, dtype, compressed=is_compressed
    )

    logger.info(
        f"Uploaded ROI details - shape: {roi_array.shape}, dtype: {roi_array.dtype}, "
        f"min: {roi_array.min()}, max: {roi_array.max()}"
    )

    # Cache the ROI for future use
    ROI_CACHE.set(roi_hash, roi_array)

    return roi_array


# --- Dependencies ---
async def ensure_active_image(image_hash: str, clim0: float, clim1: float):
    """
    A dependency that ensures the correct image with proper histogram trimming is loaded into the PROMPT_MANAGER.

    Args:
        image_hash: Hash identifier for the image
        clim0: Lower histogram boundary
        clim1: Upper histogram boundary

    Returns:
        PROMPT_MANAGER instance with the correct image loaded
    """
    global CURRENT_IMAGE_HASH, CURRENT_CLIM0, CURRENT_CLIM1

    # Check if we need to reload/reprocess the image
    needs_update = (
        image_hash != CURRENT_IMAGE_HASH
        or clim0 != CURRENT_CLIM0
        or clim1 != CURRENT_CLIM1
    )

    if needs_update:
        # Load image from cache
        image_data = IMAGE_CACHE.get(image_hash)
        if image_data is None:
            raise HTTPException(
                status_code=404, detail=f"Image {image_hash} not found in cache"
            )

        logger.info(
            f"Updating active image: hash={image_hash} (prev: {CURRENT_IMAGE_HASH}), "
            f"clim=[{clim0}, {clim1}] (prev: [{CURRENT_CLIM0}, {CURRENT_CLIM1}])"
        )

        # Apply histogram trimming
        trimmed_image = apply_histogram_trimming(image_data, clim0, clim1)

        # Set the trimmed image in the prompt manager
        PROMPT_MANAGER.set_image(trimmed_image)

        # Update global state
        CURRENT_IMAGE_HASH = image_hash
        CURRENT_CLIM0 = clim0
        CURRENT_CLIM1 = clim1

    return PROMPT_MANAGER


def check_and_reset_interactions(roi_uuid: str, roi_has_changed: bool):
    """
    Checks if interactions should be reset based on ROI UUID change or roi_has_changed flag.
    Resets interactions and updates CURRENT_ROI_UUID if necessary.

    Args:
        roi_uuid: The UUID of the current ROI
        roi_has_changed: Whether the ROI has changed since last interaction

    Returns:
        bool: True if interactions were reset, False otherwise
    """
    global CURRENT_ROI_UUID

    should_reset = False
    reset_reason = None

    if CURRENT_ROI_UUID != roi_uuid:
        should_reset = True
        reset_reason = f"ROI UUID changed from {CURRENT_ROI_UUID} to {roi_uuid}"
    elif roi_has_changed:
        should_reset = True
        reset_reason = "ROI has changed (roi_has_changed=True)"

    if should_reset:
        logger.info(f"Resetting interactions: {reset_reason}")
        PROMPT_MANAGER.session.reset_interactions()
        CURRENT_ROI_UUID = roi_uuid
        return True

    return False


@app.get("/")
async def root():
    return {"message": "Hello from Nora's segmentation server !"}


@app.get("/check_image/{image_hash}")
async def check_image_exists(image_hash: str):
    """Checks if an image with the given hash is already in the server's cache."""
    return {"exists": image_hash in IMAGE_CACHE}


@app.get("/check_image_path")
async def check_image_path_accessible(path: str):
    """Checks if an image path is accessible and readable from the server side."""
    try:
        logger.info(f"Received path: '{path}'")

        # Extract the actual path (remove the timestamp prefix if present)
        file_path = parse_filepath_with_timestamp(path)

        # Log the actual path being checked
        logger.info(f"Checking path accessibility for: '{file_path}'")
        logger.info(f"Resolved file_path object: '{file_path}'")

        # First check if path exists and is a file
        if not file_path.exists():
            logger.info(f"Path accessibility check: {file_path} -> False (does not exist)")
            return {"accessible": False}

        if not file_path.is_file():
            logger.info(f"Path accessibility check: {file_path} -> False (not a file)")
            return {"accessible": False}

        # Try to actually load the image and get data from it
        try:
            nib_img = nib.load(str(file_path))
            # Try to get the shape without loading all data into memory
            shape = nib_img.shape
            logger.info(
                f"Path accessibility check: {file_path} -> True (readable, shape: {shape})"
            )
            return {"accessible": True}
        except Exception as load_error:
            logger.warning(
                f"Path accessibility check: {file_path} -> False (cannot load with nibabel: {str(load_error)})"
            )
            return {"accessible": False}

    except ImportError:
        logger.error("Path accessibility check failed: nibabel is not installed")
        return {"accessible": False}
    except Exception as e:
        logger.error(f"Error checking path accessibility: {str(e)}")
        return {"accessible": False}


@app.get("/check_roi/{roi_hash}")
async def check_roi_exists(roi_hash: str):
    """Checks if an ROI with the given hash is already in the server's cache."""
    return {"exists": roi_hash in ROI_CACHE}


@app.post("/set_active_image/{image_hash}")
async def set_active_image(
    image_hash: str,
    clim0: float = Form(None),
    clim1: float = Form(None),
):
    """
    Sets the active image for the global prompt manager from the cache.
    Optionally applies histogram trimming if clim0 and clim1 are provided.
    """
    if clim0 is not None and clim1 is not None:
        pm = await ensure_active_image(image_hash, clim0, clim1)
        return {
            "status": "ok",
            "message": f"Active image set to {image_hash} with histogram trimming [{clim0}, {clim1}]",
        }
    else:
        # Load without histogram trimming (use full range)
        image_data = IMAGE_CACHE.get(image_hash)
        if image_data is None:
            raise HTTPException(
                status_code=404, detail=f"Image {image_hash} not found in cache"
            )
        PROMPT_MANAGER.set_image(image_data)
        global CURRENT_IMAGE_HASH, CURRENT_CLIM0, CURRENT_CLIM1
        CURRENT_IMAGE_HASH = image_hash
        CURRENT_CLIM0 = None
        CURRENT_CLIM1 = None
        return {"status": "ok", "message": f"Active image set to {image_hash} (no trimming)"}


@app.post("/reset_interactions")
async def reset_interactions(roi_uuid: str = Form(...)):
    """
    Resets all interactions if the provided ROI UUID matches the current session UUID.
    This allows users to manually clear accumulated interactions without changing the ROI.
    """
    global CURRENT_ROI_UUID

    logger.info(f"=== RESET INTERACTIONS REQUEST === uuid: {roi_uuid}")

    if CURRENT_ROI_UUID is None:
        logger.warning("No active ROI session. Nothing to reset.")
        return {"status": "warning", "message": "No active ROI session", "reset": False}

    if CURRENT_ROI_UUID != roi_uuid:
        logger.warning(
            f"UUID mismatch: requested={roi_uuid}, current={CURRENT_ROI_UUID}. Not resetting."
        )
        return {
            "status": "warning",
            "message": f"UUID mismatch. Current session UUID: {CURRENT_ROI_UUID}",
            "reset": False,
        }

    # UUIDs match - reset interactions
    logger.info(f"Resetting interactions for ROI UUID: {roi_uuid}")
    PROMPT_MANAGER.session.reset_interactions()

    return {
        "status": "ok",
        "message": f"Interactions reset for ROI UUID: {roi_uuid}",
        "reset": True,
    }


@app.post("/submit_bug_report")
async def submit_bug_report(roi_uuid: str = Form(...), report_text: str = Form(...)):
    """
    Receives and logs bug reports from users.
    The report includes the ROI UUID and a text description of the issue.
    """
    logger.info(f"=== BUG REPORT RECEIVED ===")
    logger.info(f"ROI UUID: {roi_uuid}")
    logger.info(f"Report: {report_text}")
    logger.info("=" * 50)

    return {
        "status": "ok",
        "message": "Bug report received. Thank you for your feedback!",
    }


@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(...),
    shape: str = Form(...),
    dtype: str = Form(...),
    image_hash: str = Form(...),
    absolute_path: str = Form(None),
):
    global CURRENT_IMAGE_HASH
    logger.info(f"=== UPLOAD IMAGE START === hash: {image_hash}")
    log_memory_usage("BEFORE")

    # Load image from absolute path if provided, otherwise from uploaded file
    if absolute_path:
        logger.info(f"Loading image from server path: {absolute_path}")
        try:
            # Extract the actual path (remove the timestamp prefix if present)
            file_path = parse_filepath_with_timestamp(absolute_path)
            logger.info(f"Parsed file path: '{file_path}'")

            if not file_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Image file not found: {file_path}"
                )
            if not file_path.is_file():
                raise HTTPException(
                    status_code=400, detail=f"Path is not a file: {file_path}"
                )

            # Load the image using nibabel
            nib_img = nib.load(str(file_path))
            # NiBabel returns data in (x, y, z) order, but client expects (z, y, x)
            # Transpose to match the expected array format
            img_array = nib_img.get_fdata().astype(np.float32)
            img_array = np.transpose(img_array, (2, 1, 0))  # (x, y, z) -> (z, y, x)
            logger.info(
                f"Loaded image from path - shape: {img_array.shape} (transposed to z,y,x), dtype: {img_array.dtype}"
            )

        except ImportError:
            raise HTTPException(
                status_code=500, detail="nibabel is not installed on the server"
            )
        except Exception as e:
            logger.error(f"Error loading image from path: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load image from path: {str(e)}"
            )
    else:
        logger.info("Loading image from uploaded file data")
        binary_data = await file.read()
        shape_tuple = tuple(json.loads(shape))
        img_array = deserialize_array(
            binary_data, shape_tuple, dtype, compressed=False, log_stats=False
        )

    size_mb = img_array.nbytes / (1024**2)
    logger.info(
        f"Image details - shape: {img_array.shape}, dtype: {img_array.dtype}, size: {size_mb:.2f} MB"
    )

    IMAGE_CACHE.set(image_hash, img_array)

    mean_slice = np.mean(img_array, axis=0)
    plt.imsave("mean_slice.png", mean_slice)

    # Set the image in the prompt manager
    PROMPT_MANAGER.set_image(img_array)
    CURRENT_IMAGE_HASH = image_hash

    log_memory_usage("AFTER")
    logger.info(f"=== UPLOAD IMAGE SUCCESS === hash: {image_hash}")

    return {"status": "ok"}


@app.post("/upload_roi")
async def upload_roi(
    shape: str = Form(...),
    dtype: str = Form(...),
    roi_hash: str = Form(...),
    roi_uuid: str = Form(...),
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Uploads and caches an ROI without running segmentation.
    Used to pre-cache ROI data for subsequent interaction requests.
    """
    logger.info(f"=== UPLOAD ROI START === hash: {roi_hash}, uuid: {roi_uuid}")
    log_memory_usage("BEFORE")

    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache")
        return {"status": "ok", "message": "ROI already cached", "cached": True}

    # ROI not cached, need to upload
    if file is None:
        raise HTTPException(
            status_code=400, detail="ROI not in cache and no file provided"
        )

    binary_data = await file.read()
    shape_tuple = tuple(json.loads(shape))
    is_compressed = compressed == "gzip"
    roi_array = deserialize_array(
        binary_data, shape_tuple, dtype, compressed=is_compressed
    )
    size_mb = roi_array.nbytes / (1024**2)
    logger.info(
        f"ROI details - shape: {roi_array.shape}, dtype: {roi_array.dtype}, size: {size_mb:.2f} MB, min: {roi_array.min()}, max: {roi_array.max()}"
    )

    # Cache the ROI
    ROI_CACHE.set(roi_hash, roi_array)

    log_memory_usage("AFTER")
    logger.info(f"=== UPLOAD ROI SUCCESS === hash: {roi_hash}")

    return {"status": "ok"}


@app.post("/add_roi_interaction")
async def add_roi_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    image_hash: str = Form(...),
    roi_hash: str = Form(...),
    roi_uuid: str = Form(...),
    roi_has_changed: str = Form(...),
    clim0: float = Form(...),
    clim1: float = Form(...),
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Receives an ROI mask and runs segmentation refinement on it.
    Similar to bbox/scribble interactions but uses the ROI as the only prompt.
    """
    logger.info(
        f"=== ADD ROI INTERACTION START === image: {image_hash}, roi: {roi_hash}, uuid: {roi_uuid}, changed: {roi_has_changed}, clim: [{clim0}, {clim1}]"
    )
    log_memory_usage("BEFORE")

    pm = await ensure_active_image(image_hash, clim0, clim1)

    # Check if interactions should be reset
    roi_has_changed_bool = roi_has_changed.lower() == "true"
    check_and_reset_interactions(roi_uuid, roi_has_changed_bool)

    # Load ROI from cache or upload
    roi_array = await load_roi_from_cache_or_upload(
        roi_hash, file, shape, dtype, compressed, "roi interaction"
    )

    # Set the roi in the prompt manager and run prediction
    # Note: set_segment is always called for roi_interaction endpoint since the ROI itself is the prompt
    seg_result = pm.set_segment(roi_array, run_prediction=True)
    logger.info(f"seg_result counts: {np.unique(seg_result, return_counts=True)}")
    logger.info(f"seg_result shape: {seg_result.shape}, dtype: {seg_result.dtype}")

    compressed_bin = serialize_array(seg_result, compress=True, pack_bits=True)

    log_memory_usage("AFTER")
    logger.info("=== ADD ROI INTERACTION SUCCESS ===")

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


# -- Bounding Box interaction endpoint
#
class BBoxParams(BaseModel):
    image_hash: str
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_interaction: bool


@app.post("/add_bbox_interaction")
async def add_bbox_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    params: str = Form(...),
    roi_hash: str = Form(...),
    roi_uuid: str = Form(...),
    roi_has_changed: str = Form(...),
    clim0: float = Form(...),
    clim1: float = Form(...),
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Receives bounding box corners, positive/negative interaction, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    # Parse interaction parameters from JSON string
    bbox_params = BBoxParams(**json.loads(params))
    logger.info(
        f"=== ADD BBOX INTERACTION START === image: {bbox_params.image_hash}, roi: {roi_hash}, uuid: {roi_uuid}, changed: {roi_has_changed}, clim: [{clim0}, {clim1}]"
    )
    log_memory_usage("BEFORE")

    # Ensure the correct image is active
    pm = await ensure_active_image(bbox_params.image_hash, clim0, clim1)

    # Check if interactions should be reset
    roi_has_changed_bool = roi_has_changed.lower() == "true"
    interactions_were_reset = check_and_reset_interactions(
        roi_uuid, roi_has_changed_bool
    )

    # --- Process the uploaded ROI mask ---
    # Only load and set ROI if interactions were reset (otherwise we preserve existing interactions)
    if interactions_were_reset:
        # Load ROI from cache or upload
        roi_array = await load_roi_from_cache_or_upload(
            roi_hash, file, shape, dtype, compressed, "bbox interaction"
        )

        # Set the base ROI mask in the prompt manager without running prediction yet
        pm.set_segment(roi_array, run_prediction=False)
        logger.info("Base ROI mask set for bbox interaction (interactions were reset).")
    else:
        logger.info(
            "Skipping set_segment for bbox interaction (preserving existing interactions)."
        )

    # Call the bounding box interaction method
    seg_result = pm.add_bbox_interaction(
        bbox_params.outer_point_one,
        bbox_params.outer_point_two,
        include_interaction=bbox_params.positive_interaction,
    )
    compressed_bin = serialize_array(seg_result, compress=True, pack_bits=True)

    log_memory_usage("AFTER")
    logger.info("=== ADD BBOX INTERACTION SUCCESS ===")

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


#
# -- Scribble interaction endpoint
#
class ScribbleParams(BaseModel):
    image_hash: str
    scribble_coords: list[list[float]]
    scribble_labels: list[int]
    positive_interaction: bool


@app.post("/add_scribble_interaction")
async def add_scribble_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    params: str = Form(...),
    roi_hash: str = Form(...),
    roi_uuid: str = Form(...),
    roi_has_changed: str = Form(...),
    clim0: float = Form(...),
    clim1: float = Form(...),
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Receives scribble coordinates, labels, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    # Parse interaction parameters from JSON string
    scribble_params = ScribbleParams(**json.loads(params))
    logger.info(
        f"=== ADD SCRIBBLE INTERACTION START === image: {scribble_params.image_hash}, roi: {roi_hash}, uuid: {roi_uuid}, changed: {roi_has_changed}, clim: [{clim0}, {clim1}]"
    )
    logger.info(
        f"Scribble details: {len(scribble_params.scribble_coords)} points, label: {scribble_params.scribble_labels[0]}"
    )
    log_memory_usage("BEFORE")

    # Ensure the correct image is active
    pm = await ensure_active_image(scribble_params.image_hash, clim0, clim1)

    # Check if interactions should be reset
    roi_has_changed_bool = roi_has_changed.lower() == "true"
    interactions_were_reset = check_and_reset_interactions(
        roi_uuid, roi_has_changed_bool
    )

    # --- Process the uploaded ROI mask ---
    # Only load and set ROI if interactions were reset (otherwise we preserve existing interactions)
    if interactions_were_reset:
        # Load ROI from cache or upload
        roi_array = await load_roi_from_cache_or_upload(
            roi_hash, file, shape, dtype, compressed, "scribble interaction"
        )

        # Debug visualization
        mean_roi_slice = np.mean(roi_array, axis=0)
        plt.imsave("mean_roi_slice.png", mean_roi_slice)

        # Set the base ROI mask in the prompt manager without running prediction yet
        pm.set_segment(roi_array, run_prediction=False)
        logger.info(
            "Base ROI mask set for scribble interaction (interactions were reset)."
        )
    else:
        logger.info(
            "Skipping set_segment for scribble interaction (preserving existing interactions)."
        )

    # Create a mask from scribble coordinates
    scribbles_mask = pm.create_mask_from_scribbles(
        scribble_params.scribble_coords, scribble_params.scribble_labels
    )

    mean_scribble_slice = np.mean(scribbles_mask, axis=0)
    plt.imsave("mean_scribble_slice.png", mean_scribble_slice)

    # Call the scribble interaction method
    seg_result = pm.add_scribble_interaction(
        scribbles_mask, include_interaction=scribble_params.positive_interaction
    )
    compressed_bin = serialize_array(seg_result, compress=True, pack_bits=True)

    log_memory_usage("AFTER")
    logger.info("=== ADD SCRIBBLE INTERACTION SUCCESS ===")

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


#
# -- Point interaction endpoint
#
class PointParams(BaseModel):
    image_hash: str
    point_coords: list[list[int]]  # List of [z, y, x] coordinates
    point_labels: list[int]  # List of labels (1 for positive, 0 for negative)


@app.post("/add_point_interaction")
async def add_point_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    params: str = Form(...),
    roi_hash: str = Form(...),
    roi_uuid: str = Form(...),
    roi_has_changed: str = Form(...),
    clim0: float = Form(...),
    clim1: float = Form(...),
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Receives multiple point coordinates with their labels (positive/negative), and a base ROI mask.
    Adds all points to the model, running prediction only on the last point for efficiency.
    Returns a refined mask.
    """
    # Parse interaction parameters from JSON string
    point_params = PointParams(**json.loads(params))
    logger.info(
        f"=== ADD POINT INTERACTION START === image: {point_params.image_hash}, roi: {roi_hash}, uuid: {roi_uuid}, changed: {roi_has_changed}, clim: [{clim0}, {clim1}]"
    )
    logger.info(
        f"Number of points: {len(point_params.point_coords)}, coords: {point_params.point_coords}, labels: {point_params.point_labels}"
    )
    log_memory_usage("BEFORE")

    # Validate that coords and labels have the same length
    if len(point_params.point_coords) != len(point_params.point_labels):
        raise HTTPException(
            status_code=400,
            detail=f"point_coords and point_labels must have the same length. Got {len(point_params.point_coords)} coords and {len(point_params.point_labels)} labels",
        )

    # Ensure the correct image is active
    pm = await ensure_active_image(point_params.image_hash, clim0, clim1)

    # Check if interactions should be reset
    roi_has_changed_bool = roi_has_changed.lower() == "true"
    interactions_were_reset = check_and_reset_interactions(
        roi_uuid, roi_has_changed_bool
    )

    # --- Process the uploaded ROI mask ---
    # Only load and set ROI if interactions were reset (otherwise we preserve existing interactions)
    if interactions_were_reset:
        # Load ROI from cache or upload
        roi_array = await load_roi_from_cache_or_upload(
            roi_hash, file, shape, dtype, compressed, "point interaction"
        )

        # Set the base ROI mask in the prompt manager without running prediction yet
        pm.set_segment(roi_array, run_prediction=False)
        logger.info(
            "Base ROI mask set for point interaction (interactions were reset)."
        )
    else:
        logger.info(
            "Skipping set_segment for point interaction (preserving existing interactions)."
        )

    # Add all points - run prediction only on the last one
    seg_result = None
    num_points = len(point_params.point_coords)
    for i, (point_coord, point_label) in enumerate(
        zip(point_params.point_coords, point_params.point_labels)
    ):
        is_last_point = i == num_points - 1
        include_interaction = point_label == 1  # 1 = positive, 0 = negative

        logger.info(
            f"Adding point {i+1}/{num_points}: {point_coord}, label: {point_label} (positive: {include_interaction}), run_prediction: {is_last_point}"
        )

        result = pm.add_point_interaction(
            point_coord,
            include_interaction=include_interaction,
            run_prediction=is_last_point,
        )

        if is_last_point:
            seg_result = result

    compressed_bin = serialize_array(seg_result, compress=True, pack_bits=True)

    log_memory_usage("AFTER")
    logger.info("=== ADD POINT INTERACTION SUCCESS ===")

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the remote segmentation api server."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    parser.add_argument(
        "--ssl_keyfile", type=str, default=None, help="Path to the SSL key file."
    )
    parser.add_argument(
        "--ssl_certfile",
        type=str,
        default=None,
        help="Path to the SSL certificate file.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files. Filename is auto-generated with timestamp. If not specified, only console output.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="none",
        help="Directory for persistent cache storage. Use 'none' to disable persistence (default: none - RAM only)",
    )
    args = parser.parse_args()

    # Set environment variables for configuration
    # This ensures uvicorn worker processes inherit these settings
    os.environ["SEGMENTATION_API_LOG_DIR"] = args.log_dir if args.log_dir else ""
    os.environ["SEGMENTATION_API_LOG_LEVEL"] = args.log_level
    os.environ["SEGMENTATION_API_CACHE_DIR"] = args.cache_dir

    # Configure logging with CLI arguments
    setup_logging(log_dir=args.log_dir, log_level=args.log_level)

    # Get logger after setup
    app_logger = logging.getLogger(__name__)

    # Initialize caches with optional persistence
    cache_dir = None if args.cache_dir.lower() == "none" else args.cache_dir
    init_caches(persist_dir=cache_dir)

    # Log startup configuration
    app_logger.info(f"Starting server on {args.host}:{args.port}")
    if cache_dir:
        app_logger.info(f"Cache persistence enabled: {cache_dir}")
    else:
        app_logger.info("Cache persistence disabled (RAM-only mode)")
    app_logger.info(
        f"Image cache max size: {IMAGE_CACHE.max_size_bytes / (1024**3):.2f} GB"
    )
    app_logger.info(
        f"ROI cache max size: {ROI_CACHE.max_size_bytes / (1024**3):.2f} GB"
    )
    app_logger.info(f"GPU available: {GPU_AVAILABLE}")

    # Log memory configuration
    slurm_limit = get_slurm_memory_limit()
    if slurm_limit:
        app_logger.info(
            f"Running in Slurm job with {slurm_limit / (1024**3):.2f} GB memory allocation"
        )

        # Debug: Show cgroup detection
        cgroup_usage = get_job_cgroup_memory_usage()
        if cgroup_usage:
            app_logger.info(
                f"Cgroup memory tracking enabled (current: {cgroup_usage / (1024**3):.2f} GB)"
            )
        else:
            app_logger.warning(
                "Cgroup memory tracking not available - will use process memory only"
            )
    else:
        app_logger.info("Not running in Slurm job - using system memory")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        reload=False,
        reload_excludes=["logs/*", "data/*", ".venv/*"],
        # reload_dirs=os.path.dirname(os.path.abspath(__file__))
    )


if __name__ == "__main__":
    main()
