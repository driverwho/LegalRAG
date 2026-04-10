"""Document splitting strategies."""

import logging
import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
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

        Each chunk inherits the parent document's metadata and gets additional
        chunk-level metadata:
        - chunk_id: unique identifier for this chunk
        - chunk_index: zero-based position within the parent document
        - chunk_total: total number of chunks from the parent document
        - parent_doc_id: doc_id of the source document (for reassembly)
        - char_count: character count of the chunk content

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects with enriched metadata.
        """
        if not documents:
            return []

        try:
            chunks = self._splitter.split_documents(documents)

            # Group chunks by parent doc_id to calculate chunk_total per document
            parent_groups: dict[str, list[int]] = {}
            for idx, chunk in enumerate(chunks):
                parent_id = chunk.metadata.get("doc_id", "unknown")
                parent_groups.setdefault(parent_id, []).append(idx)

            # Assign chunk-level metadata
            for parent_id, indices in parent_groups.items():
                total = len(indices)
                for position, chunk_idx in enumerate(indices):
                    chunk = chunks[chunk_idx]
                    chunk.metadata["chunk_id"] = str(uuid.uuid4())
                    chunk.metadata["chunk_index"] = position
                    chunk.metadata["chunk_total"] = total
                    chunk.metadata["parent_doc_id"] = parent_id
                    chunk.metadata["char_count"] = len(chunk.page_content)

            logger.info(
                "Split %d documents into %d chunks", len(documents), len(chunks)
            )
            return chunks
        except Exception as exc:
            logger.error("Document splitting failed: %s", exc)
            return documents
