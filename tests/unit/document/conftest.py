"""Pytest fixtures for OCR loader tests."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_ocr_result():
    """Fixture for mock OCR result."""
    return (
        [
            [None, "Line 1 text"],
            [None, "Line 2 text"],
        ],
        None,
    )


@pytest.fixture
def mock_fitz_document():
    """Fixture for mock PyMuPDF document."""
    doc = MagicMock()
    doc.page_count = 1

    page = MagicMock()
    page.get_text.return_value = "Test PDF content"
    page.get_image_info.return_value = []
    page.rect.width = 612
    page.rect.height = 792
    page.rotation = 0

    doc.__iter__ = MagicMock(return_value=iter([(0, page)]))
    return doc


@pytest.fixture
def mock_docx_document():
    """Fixture for mock python-docx document."""
    doc = MagicMock()
    doc.paragraphs = []
    doc.tables = []

    body = MagicMock()
    body.iterchildren.return_value = []
    doc.element.body = body

    return doc


@pytest.fixture
def mock_settings():
    """Fixture for mock settings."""
    settings = MagicMock()
    settings.PDF_OCR_THRESHOLD = (0.01, 0.01)
    return settings
