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
            # If there's an existing buffer, add to it, otherwise set it
            if self.target_buffer is not None:
                target_np = self.target_buffer.numpy()
                np.logical_or(target_np, mask, out=target_np)
            else:
                self.target_buffer = torch.from_numpy(mask.astype(np.uint8))

    def add_bbox_interaction(self, bbox, include_interaction=True):
        self.interactions.append(("bbox", bbox, include_interaction))
        if self.target_buffer is not None:
            z_min, z_max = bbox[0]
            y_min, y_max = bbox[1]
            x_min, x_max = bbox[2]

            value = 1 if include_interaction else 0
            self.target_buffer[z_min:z_max, y_min:y_max, x_min:x_max] = value

    def add_point_interaction(self, point, include_interaction=True, run_prediction=True):
        self.interactions.append(("point", point, include_interaction))
        if self.target_buffer is not None:
            z, y, x = int(point[0]), int(point[1]), int(point[2])

            # Create a small 3x3x3 cube around the point
            z_start, z_end = max(0, z - 1), min(self.target_buffer.shape[0], z + 2)
            y_start, y_end = max(0, y - 1), min(self.target_buffer.shape[1], y + 2)
            x_start, x_end = max(0, x - 1), min(self.target_buffer.shape[2], x + 2)

            value = 1 if include_interaction else 0
            self.target_buffer[z_start:z_end, y_start:y_end, x_start:x_end] = value
        # Note: run_prediction parameter is ignored in mock session

    def add_lasso_interaction(self, mask, include_interaction=True):
        self.interactions.append(("lasso", mask.copy(), include_interaction))
        if self.target_buffer is not None:
            target_np = self.target_buffer.numpy()
            if include_interaction:
                np.logical_or(target_np, mask, out=target_np)
            else:
                target_np[mask > 0] = 0

    def add_scribble_interaction(self, mask, include_interaction=True):
        self.interactions.append(("scribble", mask.copy(), include_interaction))
        if self.target_buffer is not None:
            target_np = self.target_buffer.numpy()
            if include_interaction:
                np.logical_or(target_np, mask, out=target_np)
            else:
                target_np[mask > 0] = 0