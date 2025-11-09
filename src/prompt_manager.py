import os

import numpy as np
import torch
from src.utils import REPO_ID, MODEL_NAME, DOWNLOAD_DIR, download_model_weights


class PromptManager:
    """
    Manages the image, target tensor, and runs inference sessions for point, bbox,
    lasso, and scribble interactions.
    """

    def __init__(self):
        self.img = None
        self.target_tensor = None

        # Track the last output to detect if client is sending back our result
        self.last_output_hash = None
        self.current_image_hash = None

        self.session = self.make_session()

    def make_session(self):
        """
        Creates an nnInteractiveInferenceSession, points it at the downloaded model.
        If MOCK_MODE is set, uses a mock session instead without GPU requirements.
        """

        if os.environ.get("MOCK_MODE", "0") == "1":

            print("Launching a mock session without GPU requirements")
            from src.mock_session import MockSession

            session = MockSession()
        else:
            from nnInteractive.inference.inference_session import (
                nnInteractiveInferenceSession,
            )
            
            model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
            
            if not os.path.exists(model_path) or not os.listdir(model_path):
                print(f"Model weights not found in '{model_path}'.")
                download_model_weights()
            else:
                print(f"Model weights found locally at '{model_path}'.")


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
        """
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(
                self.img.shape[1:], dtype=torch.uint8
            )  # Must be 3D (x, y, z)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(
                mask, run_prediction=run_prediction
            )

        if run_prediction:
            return self.target_tensor.clone().cpu().detach().numpy()

    def add_point_interaction(self, point_coordinates, include_interaction, run_prediction=True):
        """
        Process a point-based interaction (positive or negative).
        """
        self.session.add_point_interaction(
            point_coordinates, include_interaction=include_interaction, run_prediction=run_prediction
        )

        if run_prediction:
            return self.target_tensor.clone().cpu().detach().numpy()
        else:
            return None

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
        
    def create_mask_from_scribbles(self, scribble_coords, scribble_labels):
        """
        Creates a 3D mask from a list of scribble coordinates and labels.
        """
        if self.img is None:
            raise ValueError("Image not set. Cannot determine mask shape.")

        mask = np.zeros(self.img.shape[1:], dtype=np.uint8)
        for coords, label in zip(scribble_coords, scribble_labels):
            # Assuming coords are [z, y, x] and need to be integers
            z, y, x = int(coords[0]), int(coords[1]), int(coords[2])

            # Check bounds
            if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
                mask[z, y, x] = label
        return mask

    def add_scribble_interaction(self, mask, include_interaction):
        """
        Process scribble-based interaction using a 3D mask.
        """
        print("Scribble mask received with shape:", mask.shape)
        self.session.add_scribble_interaction(
            mask, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()
