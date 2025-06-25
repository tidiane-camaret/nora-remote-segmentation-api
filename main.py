from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.prompt_manager import PromptManager
import uvicorn
import argparse
import numpy as np
import json
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
    return {"message": "Hello World"}

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

    PROMPT_MANAGER.set_image(img_array)

    return {"status": "ok"}

def main():
    parser = argparse.ArgumentParser(description="Run the remote segmentation api server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()