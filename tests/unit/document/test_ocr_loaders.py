"""Unit tests for OCR document loaders.

Tests for RapidOCRPDFLoader and RapidOCRDocLoader.
Run with: python -m pytest tests/unit/document/test_ocr_loaders.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestOCRUtility(unittest.TestCase):
    """Test cases for OCR utility function (get_ocr)."""

    @patch("backend.app.core.document.ocr.RapidOCR")
    def test_get_ocr_uses_paddle_when_available(self, mock_ocr_class):
        """Test that get_ocr prefers rapidocr_paddle when available."""
        from backend.app.core.document.ocr import get_ocr

        # Simulate rapidocr_paddle being available
        mock_ocr_instance = MagicMock()
        mock_ocr_class.return_value = mock_ocr_instance

        ocr = get_ocr(use_cuda=True)

        # Verify paddle version was used with CUDA settings
        mock_ocr_class.assert_called_once_with(
            det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True
        )
        self.assertEqual(ocr, mock_ocr_instance)

    @patch("backend.app.core.document.ocr.RapidOCR")
    def test_get_ocr_fallback_to_onnxruntime(self, mock_ocr_class):
        """Test that get_ocr falls back to onnxruntime when paddle not available."""
        from backend.app.core.document.ocr import get_ocr

        # Simulate ImportError for paddle, then success for onnxruntime
        mock_ocr_class.side_effect = [
            ImportError("rapidocr_paddle not found"),
            MagicMock(),
        ]

        ocr = get_ocr(use_cuda=True)

        # Should still return an OCR instance
        self.assertIsNotNone(ocr)


class TestRapidOCRPDFLoader(unittest.TestCase):
    """Test cases for RapidOCRPDFLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_pdf_path = "test_document.pdf"

    @patch("backend.app.core.document.pdf_loader.partition_text")
    @patch("backend.app.core.document.pdf_loader.tqdm")
    @patch("backend.app.core.document.pdf_loader.fitz")
    @patch("backend.app.core.document.pdf_loader.get_settings")
    @patch("backend.app.core.document.pdf_loader.get_ocr")
    def test_basic_text_extraction_without_images(
        self, mock_get_ocr, mock_get_settings, mock_fitz, mock_tqdm, mock_partition
    ):
        """Test basic PDF text extraction when document has no images."""
        from backend.app.core.document.pdf_loader import RapidOCRPDFLoader

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.PDF_OCR_THRESHOLD = (0.01, 0.01)
        mock_get_settings.return_value = mock_settings

        mock_ocr = MagicMock()
        mock_get_ocr.return_value = mock_ocr

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 1

        # Mock page with text but no images
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is test PDF content."
        mock_page.get_image_info.return_value = []
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.rotation = 0

        mock_doc.__iter__ = Mock(return_value=iter([(0, mock_page)]))
        mock_fitz.open.return_value = mock_doc

        # Mock partition_text
        mock_element = MagicMock()
        mock_partition.return_value = [mock_element]

        # Execute
        loader = RapidOCRPDFLoader(self.test_pdf_path)
        elements = loader._get_elements()

        # Verify
        self.assertEqual(len(elements), 1)
        mock_page.get_text.assert_called_once_with("")
        mock_ocr.assert_not_called()  # OCR should not be called when no images

    @patch("backend.app.core.document.pdf_loader.partition_text")
    @patch("backend.app.core.document.pdf_loader.tqdm")
    @patch("backend.app.core.document.pdf_loader.fitz")
    @patch("backend.app.core.document.pdf_loader.get_settings")
    @patch("backend.app.core.document.pdf_loader.get_ocr")
    def test_image_ocr_extraction(
        self, mock_get_ocr, mock_get_settings, mock_fitz, mock_tqdm, mock_partition
    ):
        """Test PDF loading with OCR on embedded images."""
        from backend.app.core.document.pdf_loader import RapidOCRPDFLoader

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.PDF_OCR_THRESHOLD = (0.01, 0.01)
        mock_get_settings.return_value = mock_settings

        mock_ocr = MagicMock()
        mock_ocr.return_value = ([[None, "Extracted image text"]], None)
        mock_get_ocr.return_value = mock_ocr

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 1

        # Mock page with image
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF text content"
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.rotation = 0

        # Mock image info - large enough to trigger OCR
        mock_image_info = {
            "xref": 10,
            "bbox": (100, 100, 500, 400),  # Large image
        }
        mock_page.get_image_info.return_value = [mock_image_info]

        # Mock pixmap for image extraction
        mock_pix = MagicMock()
        mock_pix.samples = b"fake_image_data"
        mock_pix.height = 300
        mock_pix.width = 400
        mock_fitz.Pixmap.return_value = mock_pix

        mock_doc.__iter__ = Mock(return_value=iter([(0, mock_page)]))
        mock_fitz.open.return_value = mock_doc

        # Mock image processing libraries
        with patch("backend.app.core.document.pdf_loader.np") as mock_np:
            mock_np.frombuffer.return_value.reshape.return_value = MagicMock()

            with patch("backend.app.core.document.pdf_loader.cv2") as mock_cv2:
                mock_cv2.cvtColor = MagicMock(return_value=MagicMock())

                with patch("backend.app.core.document.pdf_loader.Image"):
                    mock_element = MagicMock()
                    mock_partition.return_value = [mock_element]

                    # Execute
                    loader = RapidOCRPDFLoader(self.test_pdf_path)
                    elements = loader._get_elements()

                    # Verify
                    self.assertEqual(len(elements), 1)
                    mock_ocr.assert_called_once()  # OCR should be called for image

    @patch("backend.app.core.document.pdf_loader.partition_text")
    @patch("backend.app.core.document.pdf_loader.tqdm")
    @patch("backend.app.core.document.pdf_loader.fitz")
    @patch("backend.app.core.document.pdf_loader.get_settings")
    @patch("backend.app.core.document.pdf_loader.get_ocr")
    def test_small_images_skipped_by_threshold(
        self, mock_get_ocr, mock_get_settings, mock_fitz, mock_tqdm, mock_partition
    ):
        """Test that images smaller than threshold are skipped."""
        from backend.app.core.document.pdf_loader import RapidOCRPDFLoader

        # Setup mocks with high threshold
        mock_settings = MagicMock()
        mock_settings.PDF_OCR_THRESHOLD = (0.5, 0.5)  # High threshold
        mock_get_settings.return_value = mock_settings

        mock_ocr = MagicMock()
        mock_get_ocr.return_value = mock_ocr

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 1

        # Mock page with small image
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF text"
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.rotation = 0

        # Small image (below threshold)
        mock_image_info = {
            "xref": 10,
            "bbox": (100, 100, 200, 200),  # Too small
        }
        mock_page.get_image_info.return_value = [mock_image_info]

        mock_doc.__iter__ = Mock(return_value=iter([(0, mock_page)]))
        mock_fitz.open.return_value = mock_doc

        mock_element = MagicMock()
        mock_partition.return_value = [mock_element]

        # Execute
        loader = RapidOCRPDFLoader(self.test_pdf_path)
        elements = loader._get_elements()

        # Verify - OCR should not be called for small images
        mock_ocr.assert_not_called()

    @patch("backend.app.core.document.pdf_loader.partition_text")
    @patch("backend.app.core.document.pdf_loader.tqdm")
    @patch("backend.app.core.document.pdf_loader.fitz")
    @patch("backend.app.core.document.pdf_loader.get_settings")
    @patch("backend.app.core.document.pdf_loader.get_ocr")
    def test_multiple_pages_processing(
        self, mock_get_ocr, mock_get_settings, mock_fitz, mock_tqdm, mock_partition
    ):
        """Test loading PDF with multiple pages."""
        from backend.app.core.document.pdf_loader import RapidOCRPDFLoader

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.PDF_OCR_THRESHOLD = (0.01, 0.01)
        mock_get_settings.return_value = mock_settings

        mock_ocr = MagicMock()
        mock_get_ocr.return_value = mock_ocr

        # Mock PDF document with 3 pages
        mock_doc = MagicMock()
        mock_doc.page_count = 3

        # Create 3 mock pages
        mock_pages = []
        for i in range(3):
            mock_page = MagicMock()
            mock_page.get_text.return_value = f"Page {i + 1} content"
            mock_page.get_image_info.return_value = []
            mock_page.rect.width = 612
            mock_page.rect.height = 792
            mock_page.rotation = 0
            mock_pages.append((i, mock_page))

        mock_doc.__iter__ = Mock(return_value=iter(mock_pages))
        mock_fitz.open.return_value = mock_doc

        mock_element = MagicMock()
        mock_partition.return_value = [mock_element]

        # Execute
        loader = RapidOCRPDFLoader(self.test_pdf_path)
        elements = loader._get_elements()

        # Verify all pages were processed
        self.assertEqual(len(elements), 1)
        for _, mock_page in mock_pages:
            mock_page.get_text.assert_called_once()

    @patch("backend.app.core.document.pdf_loader.partition_text")
    @patch("backend.app.core.document.pdf_loader.tqdm")
    @patch("backend.app.core.document.pdf_loader.fitz")
    @patch("backend.app.core.document.pdf_loader.get_settings")
    @patch("backend.app.core.document.pdf_loader.get_ocr")
    def test_rotated_page_handling(
        self, mock_get_ocr, mock_get_settings, mock_fitz, mock_tqdm, mock_partition
    ):
        """Test handling of rotated PDF pages."""
        from backend.app.core.document.pdf_loader import RapidOCRPDFLoader

        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.PDF_OCR_THRESHOLD = (0.01, 0.01)
        mock_get_settings.return_value = mock_settings

        mock_ocr = MagicMock()
        mock_ocr.return_value = ([[None, "Rotated image text"]], None)
        mock_get_ocr.return_value = mock_ocr

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 1

        # Mock rotated page
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF text"
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.rotation = 90  # Rotated page

        mock_image_info = {"xref": 10, "bbox": (100, 100, 500, 400)}
        mock_page.get_image_info.return_value = [mock_image_info]

        mock_pix = MagicMock()
        mock_pix.samples = b"fake_image_data"
        mock_pix.height = 300
        mock_pix.width = 400
        mock_fitz.Pixmap.return_value = mock_pix

        mock_doc.__iter__ = Mock(return_value=iter([(0, mock_page)]))
        mock_fitz.open.return_value = mock_doc

        # Mock rotation handling
        with patch("backend.app.core.document.pdf_loader.np") as mock_np:
            mock_np.frombuffer.return_value.reshape.return_value = MagicMock()

            with patch("backend.app.core.document.pdf_loader.cv2") as mock_cv2:
                mock_cv2.cvtColor = MagicMock(return_value=MagicMock())
                mock_cv2.getRotationMatrix2D = MagicMock()
                mock_cv2.warpAffine = MagicMock(return_value=MagicMock())

                with patch("backend.app.core.document.pdf_loader.Image"):
                    mock_element = MagicMock()
                    mock_partition.return_value = [mock_element]

                    # Execute
                    loader = RapidOCRPDFLoader(self.test_pdf_path)
                    elements = loader._get_elements()

                    # Verify rotation was handled
                    mock_cv2.getRotationMatrix2D.assert_called_once()


class TestRapidOCRDocLoader(unittest.TestCase):
    """Test cases for RapidOCRDocLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_docx_path = "test_document.docx"

    @patch("backend.app.core.document.docx_loader.partition_text")
    @patch("backend.app.core.document.docx_loader.tqdm")
    @patch("backend.app.core.document.docx_loader.Document")
    @patch("backend.app.core.document.docx_loader.RapidOCR")
    def test_basic_paragraph_extraction(
        self, mock_ocr_class, mock_document, mock_tqdm, mock_partition
    ):
        """Test basic DOCX paragraph extraction without images."""
        from backend.app.core.document.docx_loader import RapidOCRDocLoader

        # Setup mocks
        mock_ocr = MagicMock()
        mock_ocr_class.return_value = mock_ocr

        # Mock DOCX document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock()]
        mock_doc.tables = []

        # Mock paragraph without images
        mock_para = MagicMock()
        mock_para.text = "Test paragraph content"
        mock_para._element.xpath.return_value = []  # No images

        with patch("backend.app.core.document.docx_loader.CT_P") as mock_ct_p:
            with patch(
                "backend.app.core.document.docx_loader.Paragraph"
            ) as mock_para_class:
                mock_para_class.return_value = mock_para

                # Setup document body
                mock_body = MagicMock()
                mock_body.iterchildren.return_value = [mock_ct_p]
                mock_doc.element.body = mock_body

                mock_document.return_value = mock_doc

                mock_element = MagicMock()
                mock_partition.return_value = [mock_element]

                # Execute
                loader = RapidOCRDocLoader(self.test_docx_path)
                elements = loader._get_elements()

                # Verify
                self.assertEqual(len(elements), 1)
                mock_document.assert_called_once_with(self.test_docx_path)
                mock_ocr.assert_not_called()

    @patch("backend.app.core.document.docx_loader.partition_text")
    @patch("backend.app.core.document.docx_loader.tqdm")
    @patch("backend.app.core.document.docx_loader.Document")
    @patch("backend.app.core.document.docx_loader.RapidOCR")
    def test_image_ocr_in_paragraph(
        self, mock_ocr_class, mock_document, mock_tqdm, mock_partition
    ):
        """Test OCR on images embedded in paragraphs."""
        from backend.app.core.document.docx_loader import RapidOCRDocLoader

        # Setup mocks
        mock_ocr = MagicMock()
        mock_ocr.return_value = ([[None, "Image OCR result"]], None)
        mock_ocr_class.return_value = mock_ocr

        # Mock DOCX document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock()]
        mock_doc.tables = []

        # Mock paragraph with image
        mock_para = MagicMock()
        mock_para.text = "Paragraph with image"

        # Mock image in paragraph
        mock_pic = MagicMock()
        mock_pic.xpath.return_value = ["rId1"]
        mock_para._element.xpath.return_value = [mock_pic]

        # Mock image part
        mock_image_part = MagicMock()
        mock_image_part._blob = b"fake_image_data"
        mock_doc.part.related_parts = {"rId1": mock_image_part}

        with patch("backend.app.core.document.docx_loader.CT_P") as mock_ct_p:
            with patch(
                "backend.app.core.document.docx_loader.Paragraph"
            ) as mock_para_class:
                mock_para_class.return_value = mock_para

                mock_body = MagicMock()
                mock_body.iterchildren.return_value = [mock_ct_p]
                mock_doc.element.body = mock_body

                mock_document.return_value = mock_doc

                with patch("backend.app.core.document.docx_loader.Image") as mock_image:
                    mock_img = MagicMock()
                    mock_image.open.return_value = mock_img

                    with patch("backend.app.core.document.docx_loader.np") as mock_np:
                        mock_np.array.return_value = MagicMock()

                        mock_element = MagicMock()
                        mock_partition.return_value = [mock_element]

                        # Execute
                        loader = RapidOCRDocLoader(self.test_docx_path)
                        elements = loader._get_elements()

                        # Verify
                        self.assertEqual(len(elements), 1)
                        mock_ocr.assert_called_once()

    @patch("backend.app.core.document.docx_loader.partition_text")
    @patch("backend.app.core.document.docx_loader.tqdm")
    @patch("backend.app.core.document.docx_loader.Document")
    @patch("backend.app.core.document.docx_loader.RapidOCR")
    def test_table_extraction(
        self, mock_ocr_class, mock_document, mock_tqdm, mock_partition
    ):
        """Test extraction of text from tables."""
        from backend.app.core.document.docx_loader import RapidOCRDocLoader

        # Setup mocks
        mock_ocr = MagicMock()
        mock_ocr_class.return_value = mock_ocr

        # Mock DOCX document with table
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = [MagicMock()]

        # Mock table structure
        mock_table = MagicMock()
        mock_row = MagicMock()
        mock_cell = MagicMock()
        mock_cell_para = MagicMock()
        mock_cell_para.text = "Cell content"
        mock_cell.paragraphs = [mock_cell_para]
        mock_row.cells = [mock_cell]
        mock_table.rows = [mock_row]

        with patch("backend.app.core.document.docx_loader.CT_Tbl") as mock_ct_tbl:
            with patch(
                "backend.app.core.document.docx_loader.Table"
            ) as mock_table_class:
                mock_table_class.return_value = mock_table

                mock_body = MagicMock()
                mock_body.iterchildren.return_value = [mock_ct_tbl]
                mock_doc.element.body = mock_body

                mock_document.return_value = mock_doc

                mock_element = MagicMock()
                mock_partition.return_value = [mock_element]

                # Execute
                loader = RapidOCRDocLoader(self.test_docx_path)
                elements = loader._get_elements()

                # Verify
                self.assertEqual(len(elements), 1)

    @patch("backend.app.core.document.docx_loader.partition_text")
    @patch("backend.app.core.document.docx_loader.tqdm")
    @patch("backend.app.core.document.docx_loader.Document")
    @patch("backend.app.core.document.docx_loader.RapidOCR")
    def test_empty_document_handling(
        self, mock_ocr_class, mock_document, mock_tqdm, mock_partition
    ):
        """Test handling of empty DOCX document."""
        from backend.app.core.document.docx_loader import RapidOCRDocLoader

        # Setup mocks
        mock_ocr = MagicMock()
        mock_ocr_class.return_value = mock_ocr

        # Mock empty document
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = []

        mock_body = MagicMock()
        mock_body.iterchildren.return_value = []
        mock_doc.element.body = mock_body

        mock_document.return_value = mock_doc

        mock_element = MagicMock()
        mock_partition.return_value = [mock_element]

        # Execute
        loader = RapidOCRDocLoader(self.test_docx_path)
        elements = loader._get_elements()

        # Verify
        self.assertEqual(len(elements), 1)
        mock_ocr.assert_not_called()


class TestDocumentLoaderIntegration(unittest.TestCase):
    """Integration tests for DocumentLoader with OCR support."""

    @patch("backend.app.core.document.loader.get_ocr")
    @patch("backend.app.core.document.pdf_loader.get_ocr")
    @patch("backend.app.core.document.pdf_loader.get_settings")
    @patch("backend.app.core.document.pdf_loader.fitz")
    @patch("backend.app.core.document.pdf_loader.tqdm")
    @patch("backend.app.core.document.pdf_loader.partition_text")
    def test_pdf_loader_with_ocr_enabled(
        self,
        mock_partition,
        mock_tqdm,
        mock_fitz,
        mock_settings,
        mock_pdf_ocr,
        mock_loader_ocr,
    ):
        """Test DocumentLoader with OCR enabled for PDF files."""
        from backend.app.core.document.loader import DocumentLoader
        from langchain_core.documents import Document

        # Setup mocks
        mock_settings.return_value.PDF_OCR_THRESHOLD = (0.01, 0.01)

        mock_ocr = MagicMock()
        mock_pdf_ocr.return_value = mock_ocr
        mock_loader_ocr.return_value = mock_ocr

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 1

        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF content"
        mock_page.get_image_info.return_value = []
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.rotation = 0

        mock_doc.__iter__ = Mock(return_value=iter([(0, mock_page)]))
        mock_fitz.open.return_value = mock_doc

        # Mock partition_text to return Document
        mock_doc_result = Document(page_content="Test content", metadata={})
        mock_partition.return_value = [mock_doc_result]

        # Execute with OCR enabled
        loader = DocumentLoader(use_ocr=True)
        docs = loader._load_pdf_ocr("test.pdf")

        # Verify
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Test content")
        self.assertEqual(docs[0].metadata["file_type"], "pdf")
        self.assertEqual(docs[0].metadata["file_name"], "test.pdf")

    @patch("backend.app.core.document.loader.get_ocr")
    @patch("backend.app.core.document.docx_loader.RapidOCR")
    @patch("backend.app.core.document.docx_loader.Document")
    @patch("backend.app.core.document.docx_loader.tqdm")
    @patch("backend.app.core.document.docx_loader.partition_text")
    def test_docx_loader_with_ocr_enabled(
        self, mock_partition, mock_tqdm, mock_document, mock_ocr_class, mock_loader_ocr
    ):
        """Test DocumentLoader with OCR enabled for DOCX files."""
        from backend.app.core.document.loader import DocumentLoader
        from langchain_core.documents import Document

        # Setup mocks
        mock_ocr = MagicMock()
        mock_ocr_class.return_value = mock_ocr
        mock_loader_ocr.return_value = mock_ocr

        # Mock DOCX document
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = []

        mock_body = MagicMock()
        mock_body.iterchildren.return_value = []
        mock_doc.element.body = mock_body

        mock_document.return_value = mock_doc

        # Mock partition_text
        mock_doc_result = Document(page_content="DOCX content", metadata={})
        mock_partition.return_value = [mock_doc_result]

        # Execute with OCR enabled
        loader = DocumentLoader(use_ocr=True)
        docs = loader._load_docx_ocr("test.docx")

        # Verify
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "DOCX content")
        self.assertEqual(docs[0].metadata["file_type"], "docx")
        self.assertEqual(docs[0].metadata["file_name"], "test.docx")

    def test_loader_selection_without_ocr(self):
        """Test that standard loaders are used when OCR is disabled."""
        from backend.app.core.document.loader import DocumentLoader

        loader = DocumentLoader(use_ocr=False)

        # Verify standard loaders are in dispatch table
        self.assertEqual(loader._loaders["pdf"], loader._load_pdf)
        self.assertEqual(loader._loaders["docx"], loader._load_docx)

    def test_loader_selection_with_ocr(self):
        """Test that OCR loaders are used when OCR is enabled."""
        from backend.app.core.document.loader import DocumentLoader

        loader = DocumentLoader(use_ocr=True)

        # Verify OCR loaders are in dispatch table
        self.assertEqual(loader._loaders["pdf"], loader._load_pdf_ocr)
        self.assertEqual(loader._loaders["docx"], loader._load_docx_ocr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
