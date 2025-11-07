import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.prompt_manager import PromptManager
from src.utils import (
    ArrayCache,
    GPU_AVAILABLE,
    get_job_cgroup_memory_usage,
    get_slurm_memory_limit,
    log_memory_usage,
    parse_file_upload,
    segmentation_binary,
    setup_logging,
)

# Initialize logging at module level (will be reconfigured in main() if needed)
# This ensures logging works when uvicorn imports the module
setup_logging(log_to_file=False, log_level="INFO")
logger = logging.getLogger(__name__)

# --- Globals & App Initialization ---
IMAGE_CACHE = ArrayCache(
    max_size_bytes=1 * 1024 * 1024 * 1024, cache_name="Image"
)  # 1 GB
ROI_CACHE = ArrayCache(max_size_bytes=512 * 1024 * 1024, cache_name="ROI")  # 512 MB
PROMPT_MANAGER = PromptManager()
CURRENT_IMAGE_HASH = None  # Tracks the image hash currently loaded in PROMPT_MANAGER

app = FastAPI()

# Add CORS middleware # TODO : restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dependencies ---
async def ensure_active_image(image_hash: str):
    """
    A dependency that ensures the correct image is loaded into the PROMPT_MANAGER.
    """
    global CURRENT_IMAGE_HASH
    if image_hash != CURRENT_IMAGE_HASH:
        image_data = IMAGE_CACHE.get(image_hash)
        if image_data is None:
            raise HTTPException(
                status_code=404, detail=f"Image {image_hash} not found in cache"
            )

        logger.info(f"Switching active image from {CURRENT_IMAGE_HASH} to {image_hash}")
        PROMPT_MANAGER.set_image(image_data)
        CURRENT_IMAGE_HASH = image_hash
    return PROMPT_MANAGER


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
        import nibabel as nib

        file_path = Path(path)

        # First check if path exists and is a file
        if not file_path.exists():
            logger.info(f"Path accessibility check: {path} -> False (does not exist)")
            return {"accessible": False}

        if not file_path.is_file():
            logger.info(f"Path accessibility check: {path} -> False (not a file)")
            return {"accessible": False}

        # Try to actually load the image and get data from it
        try:
            nib_img = nib.load(str(file_path))
            # Try to get the shape without loading all data into memory
            shape = nib_img.shape
            logger.info(
                f"Path accessibility check: {path} -> True (readable, shape: {shape})"
            )
            return {"accessible": True}
        except Exception as load_error:
            logger.warning(
                f"Path accessibility check: {path} -> False (cannot load with nibabel: {str(load_error)})"
            )
            return {"accessible": False}

    except ImportError:
        logger.error("Path accessibility check failed: nibabel is not installed")
        return {"accessible": False}
    except Exception as e:
        logger.error(f"Error checking path accessibility for {path}: {str(e)}")
        return {"accessible": False}


@app.get("/check_roi/{roi_hash}")
async def check_roi_exists(roi_hash: str):
    """Checks if an ROI with the given hash is already in the server's cache."""
    return {"exists": roi_hash in ROI_CACHE}


@app.post("/set_active_image/{image_hash}")
async def set_active_image(
    image_hash: str, pm: PromptManager = Depends(ensure_active_image)
):
    """Sets the active image for the global prompt manager from the cache."""
    return {"status": "ok", "message": f"Active image set to {image_hash}"}


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
            import nibabel as nib

            file_path = Path(absolute_path)
            if not file_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Image file not found: {absolute_path}"
                )
            if not file_path.is_file():
                raise HTTPException(
                    status_code=400, detail=f"Path is not a file: {absolute_path}"
                )

            # Load the image using nibabel
            nib_img = nib.load(str(file_path))
            img_array = nib_img.get_fdata().astype(np.float32)
            logger.info(
                f"Loaded image from path - shape: {img_array.shape}, dtype: {img_array.dtype}"
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
        img_array = await parse_file_upload(file, shape, dtype)

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
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Uploads and caches an ROI without running segmentation.
    Used to pre-cache ROI data for subsequent interaction requests.
    """
    logger.info(f"=== UPLOAD ROI START === hash: {roi_hash}")
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

    roi_array = await parse_file_upload(file, shape, dtype, compressed)
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
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Receives an ROI mask and runs segmentation refinement on it.
    Similar to bbox/scribble interactions but uses the ROI as the only prompt.
    """
    logger.info(
        f"=== ADD ROI INTERACTION START === image: {image_hash}, roi: {roi_hash}"
    )
    log_memory_usage("BEFORE")

    pm = await ensure_active_image(image_hash)

    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache for roi interaction.")
        roi_array = cached_roi
    else:
        if file is None:
            raise HTTPException(
                status_code=400, detail="ROI not in cache and no file provided"
            )

        roi_array = await parse_file_upload(file, shape, dtype, compressed)
        logger.info(
            f"Uploaded ROI details - shape: {roi_array.shape}, dtype: {roi_array.dtype}, min: {roi_array.min()}, max: {roi_array.max()}"
        )
        # Cache the ROI
        ROI_CACHE.set(roi_hash, roi_array)

    # Set the roi in the prompt manager and run prediction
    seg_result = pm.set_segment(roi_array, run_prediction=True)
    logger.info(f"seg_result counts: {np.unique(seg_result, return_counts=True)}")
    logger.info(f"seg_result shape: {seg_result.shape}, dtype: {seg_result.dtype}")

    compressed_bin = segmentation_binary(seg_result, compress=True)

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
        f"=== ADD BBOX INTERACTION START === image: {bbox_params.image_hash}, roi: {roi_hash}"
    )
    log_memory_usage("BEFORE")

    # Ensure the correct image is active
    pm = await ensure_active_image(bbox_params.image_hash)

    # --- Process the uploaded ROI mask ---
    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache for bbox interaction.")
        roi_array = cached_roi
    else:
        if file is None:
            raise HTTPException(
                status_code=400, detail="ROI not in cache and no file provided"
            )
        roi_array = await parse_file_upload(file, shape, dtype, compressed)
        # Cache the ROI
        ROI_CACHE.set(roi_hash, roi_array)

    # Set the base ROI mask in the prompt manager without running prediction yet
    pm.set_segment(
        roi_array, run_prediction=False
    )  # TODO : if incoming ROI is the exact same as the one in session memory, do not set again
    logger.info("Base ROI mask set for bbox interaction.")

    # Call the bounding box interaction method
    seg_result = pm.add_bbox_interaction(
        bbox_params.outer_point_one,
        bbox_params.outer_point_two,
        include_interaction=bbox_params.positive_interaction,
    )
    compressed_bin = segmentation_binary(seg_result, compress=True)

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
        f"=== ADD SCRIBBLE INTERACTION START === image: {scribble_params.image_hash}, roi: {roi_hash}"
    )
    logger.info(
        f"Scribble details: {len(scribble_params.scribble_coords)} points, label: {scribble_params.scribble_labels[0]}"
    )
    log_memory_usage("BEFORE")

    # Ensure the correct image is active
    pm = await ensure_active_image(scribble_params.image_hash)

    # --- Process the uploaded ROI mask ---
    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache for scribble interaction.")
        roi_array = cached_roi
    else:
        if file is None:
            raise HTTPException(
                status_code=400, detail="ROI not in cache and no file provided"
            )
        roi_array = await parse_file_upload(file, shape, dtype, compressed)
        mean_roi_slice = np.mean(roi_array, axis=0)
        plt.imsave("mean_roi_slice.png", mean_roi_slice)
        # Cache the ROI
        ROI_CACHE.set(roi_hash, roi_array)

    # Set the base ROI mask in the prompt manager without running prediction yet
    pm.set_segment(roi_array, run_prediction=False)
    logger.info("Base ROI mask set for scribble interaction.")

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
    compressed_bin = segmentation_binary(seg_result, compress=True)

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
    point_coords: list[int]
    positive_interaction: bool


@app.post("/add_point_interaction")
async def add_point_interaction(
    shape: str = Form(...),
    dtype: str = Form(...),
    params: str = Form(...),
    roi_hash: str = Form(...),
    file: UploadFile = File(None),
    compressed: str = Form(None),
):
    """
    Receives point coordinates, positive/negative interaction, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    # Parse interaction parameters from JSON string
    point_params = PointParams(**json.loads(params))
    logger.info(
        f"=== ADD POINT INTERACTION START === image: {point_params.image_hash}, roi: {roi_hash}"
    )
    logger.info(
        f"Point coords: {point_params.point_coords}, positive: {point_params.positive_interaction}"
    )
    log_memory_usage("BEFORE")

    # Ensure the correct image is active
    pm = await ensure_active_image(point_params.image_hash)

    # --- Process the uploaded ROI mask ---
    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache for point interaction.")
        roi_array = cached_roi
    else:
        if file is None:
            raise HTTPException(
                status_code=400, detail="ROI not in cache and no file provided"
            )
        roi_array = await parse_file_upload(file, shape, dtype, compressed)
        # Cache the ROI
        ROI_CACHE.set(roi_hash, roi_array)

    # Set the base ROI mask in the prompt manager without running prediction yet
    pm.set_segment(roi_array, run_prediction=False)
    logger.info("Base ROI mask set for point interaction.")

    # Call the point interaction method
    seg_result = pm.add_point_interaction(
        point_params.point_coords,
        include_interaction=point_params.positive_interaction,
    )
    compressed_bin = segmentation_binary(seg_result, compress=True)

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
        "--log-file",
        action="store_true",
        help="Enable logging to file (default: console only).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging with CLI arguments
    setup_logging(log_to_file=args.log_file, log_level=args.log_level)

    # Get logger after setup
    app_logger = logging.getLogger(__name__)

    # Log startup configuration
    app_logger.info(f"Starting server on {args.host}:{args.port}")
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
        app_logger.info(f"Running in Slurm job with {slurm_limit / (1024**3):.2f} GB memory allocation")

        # Debug: Show cgroup detection
        cgroup_usage = get_job_cgroup_memory_usage()
        if cgroup_usage:
            app_logger.info(f"Cgroup memory tracking enabled (current: {cgroup_usage / (1024**3):.2f} GB)")
        else:
            app_logger.warning("Cgroup memory tracking not available - will use process memory only")
    else:
        app_logger.info("Not running in Slurm job - using system memory")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        reload=True,
        # reload_dirs=os.path.dirname(os.path.abspath(__file__))
    )


if __name__ == "__main__":
    main()
