"""Unified document loader supporting multiple file formats."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    DirectoryLoader,
)

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various file formats.

    Consolidates the previously duplicated document_loader.py and
    document_processor.py into a single implementation.
    """

    SUPPORTED_EXTENSIONS: Dict[str, str] = {
        ".txt": "text",
        ".csv": "csv",
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".xlsx": "excel",
        ".xls": "excel",
        ".md": "markdown",
    }

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
        # Strategy dispatch table — avoids long if/elif chains
        self._loaders = {
            "text": self._load_text,
            "csv": self._load_csv,
            "pdf": self._load_pdf,
            "docx": self._load_docx,
            "excel": self._load_excel,
            "markdown": self._load_markdown,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_file_type(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

    def is_supported(self, file_path: str) -> bool:
        return self.get_file_type(file_path) != "unknown"

    def load_single_file(self, file_path: str, **kwargs) -> List[Document]:
        """Load a single file into Document objects.

        Args:
            file_path: Absolute or relative path to the file.
            **kwargs: Extra arguments forwarded to the loader (e.g. CSV options).

        Returns:
            List of LangChain Document objects.
        """
        if not os.path.exists(file_path):
            logger.error("File does not exist: %s", file_path)
            return []

        file_type = self.get_file_type(file_path)
        loader_fn = self._loaders.get(file_type)

        if loader_fn is None:
            logger.warning(
                "Unsupported file type '%s' — falling back to text loader", file_path
            )
            loader_fn = self._load_text

        return loader_fn(file_path, **kwargs)

    def load_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        """Recursively load all supported files from a directory."""
        if not os.path.exists(directory_path):
            logger.error("Directory does not exist: %s", directory_path)
            return []

        exclude_patterns = exclude_patterns or []
        all_documents: List[Document] = []

        for file_path in Path(directory_path).glob(glob_pattern):
            if not file_path.is_file():
                continue
            if any(p in str(file_path) for p in exclude_patterns):
                continue
            if not self.is_supported(str(file_path)):
                logger.debug("Skipping unsupported file: %s", file_path)
                continue

            logger.info("Loading file: %s", file_path)
            docs = self.load_single_file(str(file_path))
            all_documents.extend(docs)

        logger.info(
            "Loaded %d documents from directory %s", len(all_documents), directory_path
        )
        return all_documents

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Return metadata about a file."""
        if not os.path.exists(file_path):
            return {"error": "File does not exist"}

        stat = os.stat(file_path)
        p = Path(file_path)
        return {
            "file_name": p.name,
            "file_path": str(p.absolute()),
            "file_size": stat.st_size,
            "file_type": self.get_file_type(file_path),
            "is_supported": self.is_supported(file_path),
            "extension": p.suffix.lower(),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
        }

    # ------------------------------------------------------------------
    # Private loader strategies
    # ------------------------------------------------------------------

    def _enrich_metadata(
        self, documents: List[Document], file_path: str, file_type: str, **extra
    ) -> List[Document]:
        """Attach standard metadata to every loaded document."""
        for doc in documents:
            doc.metadata.update(
                {
                    "source": file_path,
                    "file_type": file_type,
                    "file_name": Path(file_path).name,
                    **extra,
                }
            )
        return documents

    def _load_text(self, file_path: str, **kwargs) -> List[Document]:
        try:
            loader = TextLoader(file_path, encoding=self.encoding)
            docs = loader.load()
            return self._enrich_metadata(docs, file_path, "text")
        except Exception as exc:
            logger.error("Failed to load text file %s: %s", file_path, exc)
            return []

    def _load_csv(self, file_path: str, **kwargs) -> List[Document]:
        try:
            loader = CSVLoader(file_path, encoding=self.encoding, **kwargs)
            docs = loader.load()
            return self._enrich_metadata(docs, file_path, "csv")
        except Exception as exc:
            logger.error("Failed to load CSV file %s: %s", file_path, exc)
            return []

    def _load_pdf(self, file_path: str, **kwargs) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata["page_number"] = i + 1
            return self._enrich_metadata(docs, file_path, "pdf")
        except Exception as exc:
            logger.error("Failed to load PDF file %s: %s", file_path, exc)
            return []

    def _load_docx(self, file_path: str, **kwargs) -> List[Document]:
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            return self._enrich_metadata(docs, file_path, "docx")
        except Exception as exc:
            logger.error("Failed to load Word file %s: %s", file_path, exc)
            return []

    def _load_excel(self, file_path: str, **kwargs) -> List[Document]:
        try:
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()
            return self._enrich_metadata(docs, file_path, "excel")
        except Exception as exc:
            logger.error("Failed to load Excel file %s: %s", file_path, exc)
            return []

    def _load_markdown(self, file_path: str, **kwargs) -> List[Document]:
        try:
            try:
                from langchain_community.document_loaders import (
                    UnstructuredMarkdownLoader,
                )

                loader = UnstructuredMarkdownLoader(file_path)
            except ImportError:
                logger.warning(
                    "unstructured not installed — falling back to text loader for %s",
                    file_path,
                )
                loader = TextLoader(file_path, encoding=self.encoding)

            docs = loader.load()
            return self._enrich_metadata(docs, file_path, "markdown")
        except Exception as exc:
            logger.error("Failed to load Markdown file %s: %s", file_path, exc)
            return []
