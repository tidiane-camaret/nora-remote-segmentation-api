import argparse
import json
import os 

import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.prompt_manager import PromptManager, segmentation_binary

# In-memory cache for image data # TODO: implement proper caching with eviction policy
IMAGE_CACHE = {}
MAX_CACHE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
# Global prompt manager
PROMPT_MANAGER = PromptManager()

# track the current image hash
CURRENT_IMAGE_HASH = None

app = FastAPI()

# Add CORS middleware # TODO : restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




@app.get("/")
async def root():
    return {"message": "Hello from Nora's segmentation server !"}

@app.get("/check_image/{image_hash}")
async def check_image_exists(image_hash: str):
    """Checks if an image with the given hash is already in the server's cache."""
    return {"exists": image_hash in IMAGE_CACHE}

@app.post("/set_active_image/{image_hash}")
async def set_active_image(image_hash: str):
    """Sets the active image for the global prompt manager from the cache."""
    if image_hash not in IMAGE_CACHE:
        return Response(status_code=404, content=f"Image with hash {image_hash} not found in cache.")
    
    print(f"Setting active image to {image_hash}")
    PROMPT_MANAGER.set_image(IMAGE_CACHE[image_hash])
    return {"status": "ok", "message": f"Active image set to {image_hash}"}

@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(...), shape: str = Form(...), dtype: str = Form(...), image_hash: str = Form(...)
):  
    global CURRENT_IMAGE_HASH
    # Read the binary data
    binary_data = await file.read()

    # Parse the shape from JSON string
    shape_tuple = tuple(json.loads(shape))

    print(f"Recieved shape: {shape_tuple}, dtype: {dtype}")

    # Convert binary data to numpy array
    # First create array from buffer, then reshape
    img_array = np.frombuffer(binary_data, dtype=np.dtype(dtype))

    print(f"Received array of size: {img_array.shape}, dtype: {img_array.dtype}")

    img_array = img_array.reshape(shape_tuple)

    # --- Cache size management ---
    new_image_size = img_array.nbytes
    current_cache_size = sum(arr.nbytes for arr in IMAGE_CACHE.values())

    # Evict oldest images if new image exceeds cache size
    while current_cache_size + new_image_size > MAX_CACHE_SIZE_BYTES and IMAGE_CACHE:
        # FIFO eviction: remove the oldest item.
        # In Python 3.7+, dicts preserve insertion order.
        oldest_key = next(iter(IMAGE_CACHE))
        evicted_size = IMAGE_CACHE[oldest_key].nbytes
        del IMAGE_CACHE[oldest_key]
        current_cache_size -= evicted_size
        print(f"Cache limit exceeded. Evicted {oldest_key} to free up {evicted_size / (1024**2):.2f} MB.")


    IMAGE_CACHE[image_hash] = img_array
    print(f"Image {image_hash} stored in cache.")

    mean_slice = np.mean(img_array, axis=0)

    plt.imsave("mean_slice.png", mean_slice)

    print(f"Reconstructed array shape: {img_array.shape}")
    print(f"Array dtype: {img_array.dtype}")

    # Set the image in the prompt manager
    PROMPT_MANAGER.set_image(img_array)
    CURRENT_IMAGE_HASH = image_hash
    print(f"Image {image_hash} set as active.")

    return {"status": "ok"}


@app.post("/upload_roi")
async def upload_roi(
    file: UploadFile = File(...),
    shape: str = Form(...),
    dtype: str = Form(...),
    image_hash: str = Form(...)
):
    global CURRENT_IMAGE_HASH
    # Check if the correct image is loaded, if not, load it.
    if image_hash != CURRENT_IMAGE_HASH:
        if image_hash not in IMAGE_CACHE:
            return Response(status_code=404, content=f"Image with hash {image_hash} not found.")
        print(f"Switching active image from {CURRENT_IMAGE_HASH} to {image_hash}")
        PROMPT_MANAGER.set_image(IMAGE_CACHE[image_hash])
        CURRENT_IMAGE_HASH = image_hash

    # Read the binary data
    binary_data = await file.read()

    # Parse the shape from JSON string
    shape_tuple = tuple(json.loads(shape))

    # Convert binary data to numpy array
    # First create array from buffer, then reshape
    roi_array = np.frombuffer(binary_data, dtype=np.dtype(dtype))
    roi_array = roi_array.reshape(shape_tuple)

    print(f"roi min: {roi_array.min()}, max: {roi_array.max()}")
    # roi_array = roi_array.astype(np.int32)
    print(f"roi shape: {roi_array.shape}, dtype: {roi_array.dtype}")

    # Set the roi in the prompt manager
    seg_result = PROMPT_MANAGER.set_segment(roi_array, run_prediction=True)
    print(f"seg_result counts: {np.unique(seg_result, return_counts=True)}")
    print(f"seg_result shape: {seg_result.shape}, dtype: {seg_result.dtype}")
    compressed_bin = segmentation_binary(seg_result, compress=True)

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )

#
# -- Bounding Box interaction endpoint
#
class BBoxParams(BaseModel):
    image_hash: str
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_interaction: bool


@app.post("/add_bbox_interaction")
async def add_bbox_interaction(
    params: str = Form(...),
    file: UploadFile = File(...),
    shape: str = Form(...),
    dtype: str = Form(...)
):
    """
    Receives bounding box corners, positive/negative interaction, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    global CURRENT_IMAGE_HASH

    # Parse interaction parameters from JSON string
    params_dict = json.loads(params)
    bbox_params = BBoxParams(**params_dict)
    print(f"Received bbox interaction: {bbox_params}")

    # Check if the correct image is loaded, if not, load it.
    if bbox_params.image_hash != CURRENT_IMAGE_HASH:
        if bbox_params.image_hash not in IMAGE_CACHE:
            return Response(status_code=404, content=f"Image with hash {bbox_params.image_hash} not found.")
        print(f"Switching active image from {CURRENT_IMAGE_HASH} to {bbox_params.image_hash}")
        PROMPT_MANAGER.set_image(IMAGE_CACHE[bbox_params.image_hash])
        CURRENT_IMAGE_HASH = bbox_params.image_hash

    # --- Process the uploaded ROI mask ---
    binary_data = await file.read()
    shape_tuple = tuple(json.loads(shape))
    roi_array = np.frombuffer(binary_data, dtype=np.dtype(dtype)).reshape(shape_tuple)
    
    # Set the base ROI mask in the prompt manager without running prediction yet
    PROMPT_MANAGER.set_segment(roi_array, run_prediction=False)
    print("Base ROI mask set for bbox interaction.")

    # Call the bounding box interaction method
    seg_result = PROMPT_MANAGER.add_bbox_interaction(
        bbox_params.outer_point_one,
        bbox_params.outer_point_two,
        include_interaction=bbox_params.positive_interaction,
    )
    compressed_bin = segmentation_binary(seg_result, compress=True)

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
    params: str = Form(...),
    file: UploadFile = File(...),
    shape: str = Form(...),
    dtype: str = Form(...)
):
    """
    Receives scribble coordinates, labels, and a base ROI mask.
    Updates model & returns a refined mask.
    """
    global CURRENT_IMAGE_HASH

    # Parse interaction parameters from JSON string
    params_dict = json.loads(params)
    scribble_params = ScribbleParams(**params_dict)
    print(f"Received scribble interaction: {len(scribble_params.scribble_coords)} points")

    # Check if the correct image is loaded, if not, load it.
    if scribble_params.image_hash != CURRENT_IMAGE_HASH:
        if scribble_params.image_hash not in IMAGE_CACHE:
            return Response(status_code=404, content=f"Image with hash {scribble_params.image_hash} not found.")
        print(f"Switching active image from {CURRENT_IMAGE_HASH} to {scribble_params.image_hash}")
        PROMPT_MANAGER.set_image(IMAGE_CACHE[scribble_params.image_hash])
        CURRENT_IMAGE_HASH = scribble_params.image_hash

    # --- Process the uploaded ROI mask ---
    binary_data = await file.read()
    shape_tuple = tuple(json.loads(shape))
    roi_array = np.frombuffer(binary_data, dtype=np.dtype(dtype)).reshape(shape_tuple)

    mean_roi_slice = np.mean(roi_array, axis=0)
    plt.imsave("mean_roi_slice.png", mean_roi_slice)

    # Set the base ROI mask in the prompt manager without running prediction yet
    PROMPT_MANAGER.set_segment(roi_array, run_prediction=False)
    print("Base ROI mask set for scribble interaction.")

    # Create a mask from scribble coordinates
    scribbles_mask = PROMPT_MANAGER.create_mask_from_scribbles(
        scribble_params.scribble_coords, scribble_params.scribble_labels
    )

    mean_scribble_slice = np.mean(scribbles_mask, axis=0)
    plt.imsave("mean_scribble_slice.png", mean_scribble_slice)



    # Call the scribble interaction method
    seg_result = PROMPT_MANAGER.add_scribble_interaction(
        scribbles_mask, include_interaction=scribble_params.positive_interaction
    )
    compressed_bin = segmentation_binary(seg_result, compress=True)

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
    parser.add_argument("--port", type=int, default=1528, help="Port to listen on.")
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
