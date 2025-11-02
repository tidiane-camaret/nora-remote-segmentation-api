import argparse
import gzip
import json
import logging
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import psutil
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.prompt_manager import PromptManager, segmentation_binary

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("GPU monitoring not available - pynvml failed to initialize")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_memory_usage(context: str = ""):
    """Log current RAM and VRAM usage."""
    # RAM usage
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_percent = ram.percent
    
    log_msg = f"[MEMORY {context}] RAM: {ram_used_gb:.2f} GB ({ram_percent:.1f}%)"
    
    # VRAM usage (if GPU available)
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_gb = info.used / (1024**3)
            vram_total_gb = info.total / (1024**3)
            vram_percent = (info.used / info.total) * 100
            log_msg += f" | VRAM: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB ({vram_percent:.1f}%)"
        except Exception as e:
            log_msg += f" | VRAM: Error - {str(e)}"
    else:
        log_msg += " | VRAM: N/A"
    
    logger.info(log_msg)

# --- Cache Management ---
class ArrayCache:
    """Generic cache for numpy arrays (images, ROIs, etc.)"""
    def __init__(self, max_size_bytes: int, cache_name: str = "Array"):
        self._cache = OrderedDict()
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache_name = cache_name

    def get(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: np.ndarray):
        new_array_size = value.nbytes
        if new_array_size > self.max_size_bytes:
            raise ValueError(f"Array size {new_array_size} exceeds max cache size {self.max_size_bytes}")

        if key in self._cache:
            old_size = self._cache[key].nbytes
            self.current_size_bytes -= old_size

        while self.current_size_bytes + new_array_size > self.max_size_bytes:
            evicted_key, evicted_value = self._cache.popitem(last=False)
            self.current_size_bytes -= evicted_value.nbytes
            logger.info(f"[CACHE {self.cache_name}] Evicted: {evicted_key}")

        self._cache[key] = value
        self.current_size_bytes += new_array_size
        logger.info(f"[CACHE {self.cache_name}] Stored: {key} | Cache size: {self.current_size_bytes / (1024**2):.2f} MB")

    def __contains__(self, key: str) -> bool:
        return key in self._cache

# --- Globals & App Initialization ---
IMAGE_CACHE = ArrayCache(max_size_bytes=1 * 1024 * 1024 * 1024, cache_name="Image")  # 1 GB
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
            raise HTTPException(status_code=404, detail=f"Image {image_hash} not found in cache")
        
        logger.info(f"Switching active image from {CURRENT_IMAGE_HASH} to {image_hash}")
        PROMPT_MANAGER.set_image(image_data)
        CURRENT_IMAGE_HASH = image_hash
    return PROMPT_MANAGER

# --- Helper Functions ---
async def parse_file_upload(file: UploadFile, shape: str, dtype: str, compressed: str = None) -> np.ndarray:
    """Helper to parse uploaded numpy array from binary file, with optional gzip decompression."""
    binary_data = await file.read()

    # Check if data is compressed and decompress if needed
    if compressed == "gzip":
        original_size = len(binary_data)
        binary_data = gzip.decompress(binary_data)
        decompressed_size = len(binary_data)
        compression_ratio = ((1 - original_size / decompressed_size) * 100)
        print(f"Decompressed ROI: {original_size} bytes -> {decompressed_size} bytes ({compression_ratio:.2f}% compression)")

    shape_tuple = tuple(json.loads(shape))
    return np.frombuffer(binary_data, dtype=np.dtype(dtype)).reshape(shape_tuple)


@app.get("/")
async def root():
    return {"message": "Hello from Nora's segmentation server !"}

@app.get("/check_image/{image_hash}")
async def check_image_exists(image_hash: str):
    """Checks if an image with the given hash is already in the server's cache."""
    return {"exists": image_hash in IMAGE_CACHE}

@app.get("/check_roi/{roi_hash}")
async def check_roi_exists(roi_hash: str):
    """Checks if an ROI with the given hash is already in the server's cache."""
    return {"exists": roi_hash in ROI_CACHE}

@app.post("/set_active_image/{image_hash}")
async def set_active_image(image_hash: str, pm: PromptManager = Depends(ensure_active_image)):
    """Sets the active image for the global prompt manager from the cache."""
    return {"status": "ok", "message": f"Active image set to {image_hash}"}

@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(...), 
    shape: str = Form(...), 
    dtype: str = Form(...), 
    image_hash: str = Form(...)
):  
    global CURRENT_IMAGE_HASH
    logger.info(f"=== UPLOAD IMAGE START === hash: {image_hash}")
    log_memory_usage("BEFORE")

    img_array = await parse_file_upload(file, shape, dtype)

    size_mb = img_array.nbytes / (1024**2)
    logger.info(f"Image details - shape: {img_array.shape}, dtype: {img_array.dtype}, size: {size_mb:.2f} MB")
    
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
    compressed: str = Form(None)
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
        raise HTTPException(status_code=400, detail="ROI not in cache and no file provided")

    roi_array = await parse_file_upload(file, shape, dtype, compressed)
    size_mb = roi_array.nbytes / (1024**2)
    logger.info(f"ROI details - shape: {roi_array.shape}, dtype: {roi_array.dtype}, size: {size_mb:.2f} MB, min: {roi_array.min()}, max: {roi_array.max()}")

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
    compressed: str = Form(None)
):
    """
    Receives an ROI mask and runs segmentation refinement on it.
    Similar to bbox/scribble interactions but uses the ROI as the only prompt.
    """
    logger.info(f"=== ADD ROI INTERACTION START === image: {image_hash}, roi: {roi_hash}")
    log_memory_usage("BEFORE")
    
    pm = await ensure_active_image(image_hash)

    # Check if ROI is already cached
    cached_roi = ROI_CACHE.get(roi_hash)
    if cached_roi is not None:
        logger.info(f"ROI {roi_hash} found in cache for roi interaction.")
        roi_array = cached_roi
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="ROI not in cache and no file provided")
        
        roi_array = await parse_file_upload(file, shape, dtype, compressed)
        logger.info(f"Uploaded ROI details - shape: {roi_array.shape}, dtype: {roi_array.dtype}, min: {roi_array.min()}, max: {roi_array.max()}")
        # Cache the ROI
        ROI_CACHE.set(roi_hash, roi_array)

    # Set the roi in the prompt manager and run prediction
    seg_result = pm.set_segment(roi_array, run_prediction=True)
    logger.info(f"seg_result counts: {np.unique(seg_result, return_counts=True)}")
    logger.info(f"seg_result shape: {seg_result.shape}, dtype: {seg_result.dtype}")

    compressed_bin = segmentation_binary(seg_result, compress=True)

    log_memory_usage("AFTER")
    logger.info(f"=== ADD ROI INTERACTION SUCCESS ===")

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
    compressed: str = Form(None)
):
    """
    Receives bounding box corners, positive/negative interaction, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    # Parse interaction parameters from JSON string
    bbox_params = BBoxParams(**json.loads(params))
    logger.info(f"=== ADD BBOX INTERACTION START === image: {bbox_params.image_hash}, roi: {roi_hash}")
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
            raise HTTPException(status_code=400, detail="ROI not in cache and no file provided")
        roi_array = await parse_file_upload(file, shape, dtype, compressed)
        # Cache the ROI
        ROI_CACHE.set(roi_hash, roi_array)

    # Set the base ROI mask in the prompt manager without running prediction yet
    pm.set_segment(roi_array, run_prediction=False)
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
    compressed: str = Form(None)
):
    """
    Receives scribble coordinates, labels, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    # Parse interaction parameters from JSON string
    scribble_params = ScribbleParams(**json.loads(params))
    logger.info(f"=== ADD SCRIBBLE INTERACTION START === image: {scribble_params.image_hash}, roi: {roi_hash}")
    logger.info(f"Scribble details: {len(scribble_params.scribble_coords)} points, label: {scribble_params.scribble_labels[0]}")
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
            raise HTTPException(status_code=400, detail="ROI not in cache and no file provided")
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
    args = parser.parse_args()

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
