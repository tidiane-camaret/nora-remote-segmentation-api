import numpy as np
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import torch
import os
from huggingface_hub import snapshot_download
import gzip

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = ".nninteractive_weights"  # Specify the download directory


def segmentation_binary(seg_in, compress=False):
    """
    Convert a (boolean) segmentation array into packed bits and optionally compress.
    """
    seg_result = seg_in.astype(bool)  # Convert to bool type if not already
    packed_segmentation = np.packbits(seg_result, axis=None)  # Pack into 1D byte array
    packed_segmentation = packed_segmentation.tobytes()
    if compress:
        packed_segmentation = gzip.compress(packed_segmentation)
    return packed_segmentation  # Convert to bytes for transmission

class PromptManager:
    """
    Manages the image, target tensor, and runs inference sessions for point, bbox,
    lasso, and scribble interactions.
    """

    def __init__(self):
        self.img = None
        self.target_tensor = None

        self.download_weights()
        self.session = self.make_session()

    def download_weights(self):
        """
        Downloads only the files matching 'MODEL_NAME/*' into DOWNLOAD_DIR.
        """
        snapshot_download(
            repo_id=REPO_ID, allow_patterns=[f"{MODEL_NAME}/*"], local_dir=DOWNLOAD_DIR
        )

    def make_session(self):
        """
        Creates an nnInteractiveInferenceSession, points it at the downloaded model.
        """
        session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )

        # Load the trained model
        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)

        return session

    def set_image(self, input_image):
        """
        Loads the user-provided 3D image into the session, resets interactions.
        """
        self.session.reset_interactions()

        self.img = input_image[None]  # Ensure shape (1, x, y, z)
        self.session.set_image(self.img)

        print("self.img.shape:", self.img.shape)

        # Validate input dimensions
        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(
            self.img.shape[1:], dtype=torch.uint8
        )  # Must be 3D (x, y, z)
        self.session.set_target_buffer(self.target_tensor)

    def set_segment(self, mask, run_prediction=False):
        """
        Sets or resets a segmentation (mask) on the server side.
        If mask is empty, resets the session's interactions.
        """
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(
                self.img.shape[1:], dtype=torch.uint8
            )  # Must be 3D (x, y, z)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(mask, run_prediction=run_prediction)

        if run_prediction:
            return self.target_tensor.clone().cpu().detach().numpy()

    def add_point_interaction(self, point_coordinates, include_interaction):
        """
        Process a point-based interaction (positive or negative).
        """
        self.session.add_point_interaction(
            point_coordinates, include_interaction=include_interaction
        )

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_bbox_interaction(
        self, outer_point_one, outer_point_two, include_interaction
    ):
        """
        Process bounding box-based interaction.
        """
        print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)

        data = np.array([outer_point_one, outer_point_two])
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)

        bbox = [
            [int(_min[0]), int(_max[0])],
            [int(_min[1]), int(_max[1])],
            [int(_min[2]), int(_max[2])],
        ]

        # Call the session's bounding box interaction function.
        self.session.add_bbox_interaction(bbox, include_interaction=include_interaction)

        return self.target_tensor.clone().cpu().detach().numpy()

    def add_lasso_interaction(self, mask, include_interaction):
        """
        Process lasso-based interaction using a 3D mask.
        """
        print("Lasso mask received with shape:", mask.shape)
        self.session.add_lasso_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_scribble_interaction(self, mask, include_interaction):
        """
        Process scribble-based interaction using a 3D mask.
        """
        print("Scribble mask received with shape:", mask.shape)
        self.session.add_scribble_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()