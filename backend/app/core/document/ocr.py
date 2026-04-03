"""OCR utilities for document image processing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    """Get OCR instance with fallback strategy.

    Tries rapidocr_paddle first (better accuracy with GPU support),
    falls back to rapidocr_onnxruntime (lighter, CPU-only).

    Args:
        use_cuda: Whether to use CUDA for rapidocr_paddle

    Returns:
        RapidOCR instance
    """
    try:
        from rapidocr_paddle import RapidOCR

        ocr = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
        print("调用OCR识别图片成功.")
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()

    return ocr
