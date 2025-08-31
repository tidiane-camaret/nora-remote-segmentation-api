import numpy as np


class MockSession:
    """A mock session to simulate segmentation interactions."""

    def __init__(self):
        self.image = None
        self.target_buffer = None
        self.interactions = []

    def reset_interactions(self):
        self.interactions = []
        if self.image is not None:
            self.target_buffer = np.zeros(self.image.shape[1:], dtype=np.uint8)

    def set_image(self, img):
        self.image = img
        self.target_buffer = np.zeros(img.shape[1:], dtype=np.uint8)

    def set_target_buffer(self, buf):
        self.target_buffer = buf

    def add_initial_seg_interaction(self, mask, run_prediction=False):
        self.interactions.append(("seg", mask.copy()))
        if run_prediction:
            self.target_buffer = mask.astype(np.uint8)

    def add_bbox_interaction(self, bbox, include_interaction=True):
        self.interactions.append(("bbox", bbox, include_interaction))
        # Simulate a mask update (for testing)
        if self.target_buffer is not None:
            self.target_buffer[:] = 1

    def add_point_interaction(self, point, include_interaction=True):
        self.interactions.append(("point", point, include_interaction))
        # Simulate a mask update (for testing)
        if self.target_buffer is not None:
            self.target_buffer[:] = 1
            self.target_buffer[:] = 1
