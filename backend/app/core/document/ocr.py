"""OCR utilities for document image processing using PaddleOCR."""

import logging
from typing import List, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class PaddleOCRWrapper:
    """Wrapper for PaddleOCR to provide consistent interface."""

    def __init__(self, use_gpu: bool = False):
        """Initialize PaddleOCR.

        Args:
            use_gpu: Whether to use GPU (if available)
        """
        try:
            from paddleocr import PaddleOCR

            logger.info("Initializing PaddleOCR (use_gpu=%s)...", use_gpu)

            # Use device parameter for CPU/GPU selection
            device = "gpu:0" if use_gpu else "cpu"

            # Initialize PaddleOCR with CPU-optimized settings
            # Reference: https://www.paddleocr.ai/main/version3.x/pipeline_usage/OCR.html
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
                enable_mkldnn=True,  # Enable MKL-DNN acceleration for CPU
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize PaddleOCR: %s", e)
            raise RuntimeError(f"PaddleOCR initialization failed: {e}") from e

    def __call__(self, img: Union[np.ndarray, str]) -> Tuple[List, List]:
        """Run OCR on image.

        Args:
            img: Input image as numpy array or file path

        Returns:
            Tuple of (ocr_results, elapsed_times)
            ocr_results is a list of [box, (text, confidence)]
        """
        try:
            # If numpy array, save to temporary file
            if isinstance(img, np.ndarray):
                import tempfile
                import cv2

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    temp_path = f.name
                cv2.imwrite(temp_path, img)
                img_path = temp_path
            else:
                img_path = img
                temp_path = None

            # Run OCR using PaddleOCR 3.x API
            # Reference: https://www.paddleocr.ai/main/version3.x/pipeline_usage/OCR.html#22-python
            result = self.ocr.ocr(img_path)

            # Clean up temp file if created
            if temp_path:
                import os

                os.unlink(temp_path)

            # Convert result to expected format
            # PaddleOCR 3.x returns: [[page_results]] where page_results is a list of:
            # [[bbox, (text, confidence)], ...]
            # or None if no text detected
            if result and len(result) > 0:
                page_result = result[0]  # Get first page result
                if page_result is None:
                    return [], []

                # page_result is already in the format we need: [[bbox, (text, confidence)], ...]
                return page_result, []
            else:
                return [], []

        except Exception as e:
            logger.error("OCR processing failed: %s", e)
            return [], []


def get_ocr(use_cuda: bool = False):
    """Get OCR instance.

    Args:
        use_cuda: Whether to use CUDA (not used, kept for compatibility)

    Returns:
        PaddleOCRWrapper instance
    """
    return PaddleOCRWrapper(use_gpu=use_cuda)
