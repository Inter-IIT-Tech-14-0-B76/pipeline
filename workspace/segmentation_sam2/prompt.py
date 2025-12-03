import torch
import numpy as np
import cv2
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Use the unified model cache to obtain SAM models instead of building directly
from helpers.model_cache import get_model_cache

OUTPUT_MASK = "outputs/mask.png"


class Segmentation:
    def __init__(
        self,
        config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_path="sam2.1_hiera_large.pt",
        device=None,
    ):
        """
        SAM2 pipeline (point, box, lasso), matching the MobileSAM API.
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Obtain SAM2 model from the unified model cache (cached)
        self.model = get_model_cache().get_sam2(config_path, checkpoint_path)

        # Wrap with the SAM2 Image Predictor
        self.predictor = SAM2ImagePredictor(self.model)

        self.prev_mask = None
        self.image_shape = None

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------
    def set_image(self, image_np):
        """
        Store input image & precompute SAM2 embeddings
        """
        self.image_shape = image_np.shape[:2]
        self.predictor.set_image(image_np)

    # ------------------------------------------------------------
    # Invert mask
    # ------------------------------------------------------------
    def invert_selection(self):
        if self.prev_mask is not None:
            self.prev_mask = 1 - self.prev_mask

    # ------------------------------------------------------------
    # POINT PROMPT
    # ------------------------------------------------------------
    def point_prompt(self, point, is_foreground=True):
        x, y = point
        pts = np.array([[x, y]], dtype=np.float32)
        lbl = np.array([1 if is_foreground else 0], dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            multimask_output=False,
            normalize_coords=True,
        )

        self.prev_mask = masks[0].astype(np.uint8)
        Image.fromarray(self.prev_mask * 255).save(OUTPUT_MASK)

    # ------------------------------------------------------------
    # BOX PROMPT
    # ------------------------------------------------------------
    def box_prompt(self, box):
        masks, scores, logits = self.predictor.predict(
            box=np.array(box, dtype=np.float32),
            multimask_output=False,
            normalize_coords=True,
        )

        self.prev_mask = masks[0].astype(np.uint8)
        Image.fromarray(self.prev_mask * 255).save(OUTPUT_MASK)

    # ------------------------------------------------------------
    # Rich prompting: sample interior points
    # ------------------------------------------------------------
    def _sample_interior_points(self, mask, n_points=10, margin=5):
        H, W = mask.shape

        kernel = np.ones((margin, margin), np.uint8)
        safe_mask = cv2.erode(mask, kernel)

        ys, xs = np.where(safe_mask > 0)
        if len(xs) == 0:
            return None, None

        idx = np.random.choice(len(xs), size=min(n_points, len(xs)), replace=False)
        pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

        labels = np.ones(len(pts), dtype=np.int32)
        return pts, labels

    # ------------------------------------------------------------
    # LASSO PROMPT (polygon)
    # ------------------------------------------------------------
    def lasso_prompt(self, polygon_points):
        H, W = self.image_shape

        # 1. polygon → full-res mask
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(poly_mask, [pts], 1)

        # 3. sample interior pts
        input_pts, labels = self._sample_interior_points(
            poly_mask, n_points=10, margin=8
        )
        if input_pts is None:
            return None

        # 4. bounding box
        ys, xs = np.where(poly_mask > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        # 5. predict using both prompts
        masks, scores, logits = self.predictor.predict(
            point_coords=input_pts,
            point_labels=labels,
            box=box,
            multimask_output=False,
            normalize_coords=True,
        )

        self.prev_mask = masks[0].astype(np.uint8)
        Image.fromarray(self.prev_mask * 255).save(OUTPUT_MASK)

