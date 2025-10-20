import os
from huggingface_hub import snapshot_download

# Shared constants for model and download paths
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = ".nninteractive_weights"

def download_model_weights():
    """
    Downloads model weights from Hugging Face Hub if they are not already present.
    """
    print(f"Checking for model weights: {MODEL_NAME}")

    # The download directory is relative to the project root
    model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)

    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"Model weights found in '{model_path}'. Skipping download.")
        return

    print(f"Downloading model from '{REPO_ID}' to '{DOWNLOAD_DIR}'...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR,
            local_dir_use_symlinks=False,
        )
        print("Model download complete.")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        exit(1)

if __name__ == "__main__":
    # Allows the script to be run directly to download weights.
    download_model_weights()