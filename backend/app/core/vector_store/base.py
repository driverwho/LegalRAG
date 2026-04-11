"""Abstract base class for vector store implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Interface that all vector store backends must implement.

    This abstraction allows swapping ChromaDB for Milvus, Pinecone, etc.
    without changing the retrieval or API layers.
    """

    @abstractmethod
    def add_documents(
        self, documents: List[Document], collection_name: Optional[str] = None
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: LangChain Document objects to store.
            collection_name: Target collection (uses default if None).
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
        collection_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Similarity search returning (document, score) pairs.

        Args:
            query: Query text.
            k: Number of results to return.
            filter_dict: Optional metadata filters.
            collection_name: Target collection (uses default if None).

        Returns:
            List of (Document, similarity_score) tuples.
        """
        ...

    @abstractmethod
    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get collection metadata and statistics.

        Args:
            collection_name: Target collection (uses default if None).

        Returns:
            Dictionary with collection stats (document_count, etc.).
        """
        ...

    @abstractmethod
    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete all documents in a collection.

        Args:
            collection_name: Target collection (uses default if None).
        """
        ...

    @abstractmethod
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with basic stats.

        Returns:
            List of dicts with keys: name, document_count.
        """
        ...

    @abstractmethod
    def get_documents(
        self,
        collection_name: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
        keyword: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get documents from a collection with pagination.

        Args:
            collection_name: Target collection (uses default if None).
            offset: Number of documents to skip.
            limit: Maximum number of documents to return.
            keyword: Optional keyword to filter document content.

        Returns:
            Dict with keys: documents (list), total (int), offset, limit.
        """
        ...

    @abstractmethod
    def get_document(
        self, doc_id: str, collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a single document by ID.

        Args:
            doc_id: Document ID.
            collection_name: Target collection (uses default if None).

        Returns:
            Dict with keys: id, content, metadata — or None if not found.
        """
        ...

    @abstractmethod
    def delete_documents(
        self, ids: List[str], collection_name: Optional[str] = None
    ) -> int:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
            collection_name: Target collection (uses default if None).

        Returns:
            Number of documents deleted.
        """
        ...

    @abstractmethod
    def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Update a document's content and/or metadata.

        Args:
            doc_id: Document ID to update.
            content: New content (None to keep existing).
            metadata: New metadata dict (None to keep existing).
            collection_name: Target collection (uses default if None).

        Returns:
            True if updated successfully, False if not found.
        """
        ...
