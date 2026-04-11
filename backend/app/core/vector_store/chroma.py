"""ChromaDB implementation of the BaseVectorStore interface."""

import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.app.core.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by ChromaDB with LangChain integration."""

    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self._vectorstore: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> chromadb.ClientAPI:
        return chromadb.PersistentClient(path=self.persist_directory)

    def _collection_exists(self, name: str) -> bool:
        client = self._get_client()
        return name in [c.name for c in client.list_collections()]

    def _load_vectorstore(self, name: str) -> Optional[Chroma]:
        """Load an existing collection as a LangChain Chroma vectorstore."""
        if not self._collection_exists(name):
            return None
        return Chroma(
            collection_name=name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    @staticmethod
    def _sanitize_documents(documents: List[Document]) -> List[Document]:
        """Re-create Document objects to avoid missing-id errors."""
        return [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]

    # ------------------------------------------------------------------
    # BaseVectorStore implementation
    # ------------------------------------------------------------------

    def add_documents(
        self, documents: List[Document], collection_name: Optional[str] = None
    ) -> None:
        if not documents:
            logger.warning("No documents to add — skipping")
            return

        target = collection_name or self.collection_name

        try:
            if self._collection_exists(target):
                vs = Chroma(
                    collection_name=target,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                )
                vs.add_documents(documents)
                logger.info(
                    "Appended %d documents to existing collection '%s'",
                    len(documents),
                    target,
                )
            else:
                vs = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=target,
                    persist_directory=self.persist_directory,
                )
                logger.info(
                    "Created collection '%s' with %d documents", target, len(documents)
                )

            self._vectorstore = vs
            self.collection_name = target

        except Exception as exc:
            # Handle the known LangChain Document-ID bug
            if "'Document' object has no" in str(exc):
                logger.warning("Document ID issue detected — sanitizing and retrying")
                sanitized = self._sanitize_documents(documents)
                try:
                    if self._collection_exists(target):
                        vs = Chroma(
                            collection_name=target,
                            embedding_function=self.embeddings,
                            persist_directory=self.persist_directory,
                        )
                        vs.add_documents(sanitized)
                    else:
                        vs = Chroma.from_documents(
                            documents=sanitized,
                            embedding=self.embeddings,
                            collection_name=target,
                            persist_directory=self.persist_directory,
                        )
                    self._vectorstore = vs
                    self.collection_name = target
                    logger.info(
                        "Retry succeeded — added %d documents to '%s'",
                        len(sanitized),
                        target,
                    )
                except Exception as retry_exc:
                    logger.error("Retry failed: %s", retry_exc)
                    raise retry_exc from exc
            else:
                logger.error("Failed to add documents: %s", exc)
                raise

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
        collection_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        target = collection_name or self.collection_name
        vs = self._load_vectorstore(target)
        if vs is None:
            logger.warning(
                "Collection '%s' does not exist — returning empty results", target
            )
            return []

        try:
            if filter_dict:
                results = vs.similarity_search_with_score(
                    query=query, k=k, filter=filter_dict
                )
            else:
                results = vs.similarity_search_with_score(query=query, k=k)

            logger.info(
                "Search '%s' in '%s' returned %d results",
                query[:50],
                target,
                len(results),
            )
            return results
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            return []

    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        target = collection_name or self.collection_name
        info: Dict[str, Any] = {
            "persist_directory": self.persist_directory,
            "collection_name": target,
            "is_initialized": False,
            "document_count": 0,
        }

        try:
            client = self._get_client()
            if self._collection_exists(target):
                collection = client.get_collection(name=target)
                info["document_count"] = collection.count()
                info["is_initialized"] = True
        except Exception as exc:
            logger.error("Failed to get collection info: %s", exc)
            info["error"] = str(exc)

        return info

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        target = collection_name or self.collection_name
        try:
            client = self._get_client()
            try:
                client.delete_collection(target)
                logger.info("Deleted collection '%s'", target)
            except ValueError:
                logger.info(
                    "Collection '%s' does not exist — nothing to delete", target
                )
            self._vectorstore = None
        except Exception as exc:
            logger.error("Failed to clear collection: %s", exc)

    # ------------------------------------------------------------------
    # Knowledge-base management (CRUD on individual documents)
    # ------------------------------------------------------------------

    def list_collections(self) -> List[Dict[str, Any]]:
        try:
            client = self._get_client()
            result = []
            for col in client.list_collections():
                try:
                    count = client.get_collection(name=col.name).count()
                except Exception:
                    count = 0
                result.append({"name": col.name, "document_count": count})
            return result
        except Exception as exc:
            logger.error("Failed to list collections: %s", exc)
            return []

    def get_documents(
        self,
        collection_name: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
        keyword: Optional[str] = None,
    ) -> Dict[str, Any]:
        target = collection_name or self.collection_name
        empty = {"documents": [], "total": 0, "offset": offset, "limit": limit}

        try:
            client = self._get_client()
            if not self._collection_exists(target):
                return empty

            collection = client.get_collection(name=target)
            total = collection.count()
            if total == 0:
                return empty

            if keyword:
                data = collection.get(
                    where_document={"$contains": keyword},
                    include=["documents", "metadatas"],
                )
                all_ids = data.get("ids", [])
                all_docs = data.get("documents", [])
                all_metas = data.get("metadatas", [])
                filtered_total = len(all_ids)

                page_ids = all_ids[offset : offset + limit]
                page_docs = all_docs[offset : offset + limit]
                page_metas = all_metas[offset : offset + limit]

                documents = []
                for i, doc_id in enumerate(page_ids):
                    documents.append({
                        "id": doc_id,
                        "content": page_docs[i] if page_docs else "",
                        "metadata": page_metas[i] if page_metas else {},
                    })
                return {
                    "documents": documents,
                    "total": filtered_total,
                    "offset": offset,
                    "limit": limit,
                }

            data = collection.get(
                include=["documents", "metadatas"],
                offset=offset,
                limit=limit,
            )
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            documents = []
            for i, doc_id in enumerate(ids):
                documents.append({
                    "id": doc_id,
                    "content": docs[i] if docs else "",
                    "metadata": metas[i] if metas else {},
                })

            return {
                "documents": documents,
                "total": total,
                "offset": offset,
                "limit": limit,
            }
        except Exception as exc:
            logger.error("Failed to get documents: %s", exc)
            return empty

    def get_document(
        self, doc_id: str, collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        target = collection_name or self.collection_name
        try:
            client = self._get_client()
            if not self._collection_exists(target):
                return None

            collection = client.get_collection(name=target)
            data = collection.get(
                ids=[doc_id], include=["documents", "metadatas"]
            )

            if not data["ids"]:
                return None

            return {
                "id": data["ids"][0],
                "content": data["documents"][0] if data["documents"] else "",
                "metadata": data["metadatas"][0] if data["metadatas"] else {},
            }
        except Exception as exc:
            logger.error("Failed to get document '%s': %s", doc_id, exc)
            return None

    def delete_documents(
        self, ids: List[str], collection_name: Optional[str] = None
    ) -> int:
        target = collection_name or self.collection_name
        try:
            client = self._get_client()
            if not self._collection_exists(target):
                return 0

            collection = client.get_collection(name=target)
            existing = collection.get(ids=ids, include=[])
            found_ids = existing["ids"]
            if not found_ids:
                return 0

            collection.delete(ids=found_ids)
            logger.info(
                "Deleted %d documents from '%s'", len(found_ids), target
            )
            return len(found_ids)
        except Exception as exc:
            logger.error("Failed to delete documents: %s", exc)
            return 0

    def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        target = collection_name or self.collection_name
        try:
            client = self._get_client()
            if not self._collection_exists(target):
                return False

            collection = client.get_collection(name=target)
            existing = collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if not existing["ids"]:
                return False

            update_kwargs: Dict[str, Any] = {"ids": [doc_id]}
            if content is not None:
                update_kwargs["documents"] = [content]
            if metadata is not None:
                update_kwargs["metadatas"] = [metadata]

            # Re-embed if content changed
            if content is not None:
                embedding = self.embeddings.embed_documents([content])
                update_kwargs["embeddings"] = embedding

            collection.update(**update_kwargs)
            logger.info("Updated document '%s' in '%s'", doc_id, target)
            return True
        except Exception as exc:
            logger.error("Failed to update document '%s': %s", doc_id, exc)
            return False
