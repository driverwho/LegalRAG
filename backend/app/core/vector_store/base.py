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
