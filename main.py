from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.prompt_manager import PromptManager, segmentation_binary
import uvicorn
import argparse
import numpy as np
import json
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware # TODO : restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

PROMPT_MANAGER = PromptManager()

@app.get("/")
async def root():
    return {"message": "Hello from Nora's segmentation server!"}

@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(...),
    shape: str = Form(...),
    dtype: str = Form(...)
):
    # Read the binary data
    binary_data = await file.read()
    
    # Parse the shape from JSON string
    shape_tuple = tuple(json.loads(shape))
    
    # Convert binary data to numpy array
    # First create array from buffer, then reshape
    img_array = np.frombuffer(binary_data, dtype=np.dtype(dtype))
    img_array = img_array.reshape(shape_tuple)
    
    print(f"Reconstructed array shape: {img_array.shape}")
    print(f"Array dtype: {img_array.dtype}")

    # Set the image in the prompt manager
    PROMPT_MANAGER.set_image(img_array)

    return {"status": "ok"}

@app.post("/upload_roi")
async def upload_roi(
    file: UploadFile = File(...),
    shape: str = Form(...),
    dtype: str = Form(...)
):
    # Read the binary data
    binary_data = await file.read()
    
    # Parse the shape from JSON string
    shape_tuple = tuple(json.loads(shape))
    
    # Convert binary data to numpy array
    # First create array from buffer, then reshape
    roi_array = np.frombuffer(binary_data, dtype=np.dtype(dtype))
    roi_array = roi_array.reshape(shape_tuple)
    
    print(f"roi min: {roi_array.min()}, max: {roi_array.max()}")
    #roi_array = roi_array.astype(np.int32) 
    print(f"roi shape: {roi_array.shape}, dtype: {roi_array.dtype}")


    # Set the image in the prompt manager
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
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_click: bool

@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    """
    Receives bounding box corners + positive/negative. Updates model & returns a mask.
    """
    print(f"Received bbox interaction: {params}")
    

    # Set the image in the prompt manager
    seg_result = PROMPT_MANAGER.add_bbox_interaction(
        params.outer_point_one,
        params.outer_point_two,
        include_interaction=params.positive_click,
    )
    compressed_bin = segmentation_binary(seg_result, compress=True)

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )

def main():
    parser = argparse.ArgumentParser(description="Run the remote segmentation api server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    parser.add_argument("--ssl_keyfile", type=str, default=None, help="Path to the SSL key file.")
    parser.add_argument("--ssl_certfile", type=str, default=None, help="Path to the SSL certificate file.")
    args = parser.parse_args()

    uvicorn.run("main:app", 
                host=args.host, 
                port=args.port, 
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                reload=True)


if __name__ == "__main__":
    main()