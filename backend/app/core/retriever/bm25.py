"""BM25 retriever for keyword-based search."""

import logging
from typing import List, Tuple, Optional, Dict, Any

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based keyword retriever with Chinese tokenization support.

    Uses jieba for Chinese word segmentation and BM25Okapi for ranking.
    """

    def __init__(self, corpus: Optional[List[Document]] = None):
        """Initialize BM25 retriever.

        Args:
            corpus: Optional initial corpus of documents
        """
        self.corpus: List[Document] = corpus or []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if corpus:
            self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text using jieba.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Use jieba for Chinese word segmentation
        tokens = jieba.lcut(text.lower())
        # Filter out single-character tokens and punctuation
        tokens = [t for t in tokens if len(t) > 1 and t.isalnum()]
        return tokens

    def _build_index(self):
        """Build BM25 index from corpus."""
        if not self.corpus:
            logger.warning("Empty corpus, BM25 index not built")
            return

        logger.info("Building BM25 index for %d documents", len(self.corpus))

        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self.corpus
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 index built successfully")

    def add_documents(self, documents: List[Document]):
        """Add documents to the corpus and rebuild index.

        Args:
            documents: Documents to add
        """
        self.corpus.extend(documents)
        self._build_index()

    def search(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for documents using BM25.

        Args:
            query: Search query
            k: Number of results to return
            collection_name: Optional collection filter
            filter_dict: Optional metadata filters

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if not self.bm25 or not self.corpus:
            logger.warning("BM25 index not initialized, returning empty results")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)
        print("BM25 query tokens for '%s': %s", query[:50], tokenized_query)

        if not tokenized_query:
            logger.warning("Query tokenization resulted in empty tokens")
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Apply filters
        filtered_indices = []
        for idx, doc in enumerate(self.corpus):
            # Collection filter
            if collection_name and doc.metadata.get("collection") != collection_name:
                continue

            # Metadata filters
            if filter_dict:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue

            filtered_indices.append(idx)

        # Get top-k from filtered results
        if not filtered_indices:
            return []

        filtered_scores = [(idx, scores[idx]) for idx in filtered_indices]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for idx, score in filtered_scores[:k]:
            results.append((self.corpus[idx], float(score)))

        logger.info(
            "BM25 search for '%s' returned %d results (from %d filtered)",
            query[:50], len(results), len(filtered_indices)
        )

        return results

    def get_corpus_size(self) -> int:
        """Get the number of documents in the corpus."""
        return len(self.corpus)

    def clear(self):
        """Clear the corpus and index."""
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        logger.info("BM25 index cleared")


class HybridBM25Retriever(BM25Retriever):
    """BM25 retriever that syncs with a vector store.

    Automatically loads corpus from vector store collections.
    """

    def __init__(self, vector_store):
        """Initialize hybrid retriever.

        Args:
            vector_store: Vector store to sync corpus from
        """
        super().__init__()
        self.vector_store = vector_store
        self._sync_from_vector_store()

    def _sync_from_vector_store(self):
        """Load all documents from vector store into BM25 corpus."""
        try:
            # Get all collections
            collections = self.vector_store.list_collections()

            all_documents = []
            for collection_info in collections:
                collection_name = collection_info["name"]

                # Get all documents from this collection
                result = self.vector_store.get_documents(
                    collection_name=collection_name,
                    offset=0,
                    limit=10000,  # Adjust based on your corpus size
                )

                # Convert to Document objects
                for doc_data in result["documents"]:
                    doc = Document(
                        page_content=doc_data["content"],
                        metadata={
                            **doc_data["metadata"],
                            "collection": collection_name,
                        }
                    )
                    all_documents.append(doc)

            logger.info(
                "Synced %d documents from %d collections to BM25 index",
                len(all_documents), len(collections)
            )

            # Build index
            self.corpus = all_documents
            self._build_index()

        except Exception as exc:
            logger.error("Failed to sync from vector store: %s", exc)

    def refresh(self):
        """Refresh BM25 index from vector store."""
        self.clear()
        self._sync_from_vector_store()
