"""Document splitting strategies."""

import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Configurable document chunking using RecursiveCharacterTextSplitter.

    The separator list is tuned for Chinese + English mixed content.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """
        if not documents:
            return []

        try:
            chunks = self._splitter.split_documents(documents)
            logger.info(
                "Split %d documents into %d chunks", len(documents), len(chunks)
            )
            return chunks
        except Exception as exc:
            logger.error("Document splitting failed: %s", exc)
            return documents
