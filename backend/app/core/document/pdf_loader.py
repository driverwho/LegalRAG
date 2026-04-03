"""PDF loader with OCR support for image extraction."""

import logging
from typing import List

import cv2
import numpy as np
import tqdm
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from PIL import Image

from backend.app.config.settings import get_settings
from backend.app.core.document.ocr import get_ocr

logger = logging.getLogger(__name__)


class RapidOCRPDFLoader(UnstructuredFileLoader):
    """PDF loader that extracts text from both text content and images using OCR.

    Extends UnstructuredFileLoader to provide enhanced PDF loading with
    image text extraction via RapidOCR.
    """

    def _get_elements(self) -> List:
        """Extract text elements from PDF including OCR from images."""

        def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
            """Rotate image by given angle.

            Args:
                img: Input image array
                angle: Rotation angle in degrees (positive = counter-clockwise)

            Returns:
                Rotated image array
            """
            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            # Get rotation matrix
            # param1: rotation center
            # param2: rotation angle (positive = counter-clockwise)
            # param3: scale factor
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # Calculate new image boundaries
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # Adjust rotation matrix for translation
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img

        def pdf2text(filepath: str) -> str:
            """Extract text from PDF including OCR from embedded images.

            Args:
                filepath: Path to PDF file

            Returns:
                Combined text content from PDF text and image OCR
            """
            import fitz  # PyMuPDF

            settings = get_settings()
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(
                total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0"
            )
            for i, page in enumerate(doc):
                b_unit.set_description(
                    "RapidOCRPDFLoader context page index: {}".format(i)
                )
                b_unit.refresh()

                # Extract text content
                text = page.get_text("")
                resp += text + "\n"

                # Process images on the page
                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    xref = img.get("xref")
                    if not xref:
                        continue

                    bbox = img["bbox"]
                    # Skip images smaller than threshold
                    width_ratio = (bbox[2] - bbox[0]) / page.rect.width
                    height_ratio = (bbox[3] - bbox[1]) / page.rect.height

                    if (
                        width_ratio < settings.PDF_OCR_THRESHOLD[0]
                        or height_ratio < settings.PDF_OCR_THRESHOLD[1]
                    ):
                        continue

                    # Extract and process image
                    try:
                        pix = fitz.Pixmap(doc, xref)

                        # Convert CMYK or other color spaces to RGB
                        if pix.n > 4:
                            # CMYK: convert to RGB first
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Convert to PIL Image
                        if pix.n == 4:
                            # RGBA
                            img = Image.frombytes(
                                "RGBA", [pix.width, pix.height], pix.samples
                            )
                            img = img.convert("RGB")
                        elif pix.n == 1:
                            # Grayscale
                            img = Image.frombytes(
                                "L", [pix.width, pix.height], pix.samples
                            )
                            img = img.convert("RGB")
                        else:
                            # RGB
                            img = Image.frombytes(
                                "RGB", [pix.width, pix.height], pix.samples
                            )

                        # Convert PIL Image to numpy array (BGR format for OpenCV)
                        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                        # Handle page rotation if needed
                        if int(page.rotation) != 0:
                            img_array = rotate_img(
                                img=img_array, angle=360 - page.rotation
                            )

                        # Perform OCR on image
                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result) + "\n"
                    except Exception as img_exc:
                        logger.warning(
                            f"Failed to process image on page {i}: {img_exc}"
                        )
                        continue

                # Update progress
                b_unit.update(1)

            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)
