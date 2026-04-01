"""
向量存储模块
专门处理向量数据库的存储和检索功能
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# LangChain imports
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb

logger = logging.getLogger(__name__)


class VectorStorage:
    """向量存储类，专门处理ChromaDB的存储和检索"""
    
    def __init__(self, 
                 collection_name: str,
                 embeddings: Embeddings,
                 persist_directory: str):
        """
        初始化向量存储
        
        Args:
            collection_name: ChromaDB 集合名称
            embeddings: 嵌入模型实例
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None

    def add_documents(self, documents: List[Document], collection_name: str = None):
        """
        将文档添加到ChromaDB数据库
        
        Args:
            documents: 文档列表
            collection_name: 集合名称（可选，覆盖默认）
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return
        
        target_collection = collection_name or self.collection_name
        
        try:
            # 检查集合是否存在
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [c.name for c in client.list_collections()]
            
            if self.vectorstore is None or self.collection_name != target_collection:
                # 初始化 vectorstore
                if target_collection in collections:
                    logger.info(f"加载现有集合: {target_collection}")
                else:
                    logger.info(f"集合不存在，将创建新集合: {target_collection}")
                
                # 策略：如果集合存在，用 Chroma 加载，然后 add_documents()
                # 如果集合不存在，用 Chroma.from_documents()
                
                if target_collection in collections:
                    self.vectorstore = Chroma(
                        collection_name=target_collection,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    self.collection_name = target_collection
                    # 追加数据
                    self.vectorstore.add_documents(documents)
                    logger.info(f"成功向现有集合 '{target_collection}' 追加 {len(documents)} 条文档")
                else:
                    # 集合不存在，创建并插入
                    self.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=target_collection,
                        persist_directory=self.persist_directory
                    )
                    self.collection_name = target_collection
                    logger.info(f"成功创建集合 '{target_collection}' 并插入 {len(documents)} 条文档")
            else:
                # vectorstore 已初始化且集合名称匹配，直接追加
                self.vectorstore.add_documents(documents)
                logger.info(f"成功向当前集合 '{target_collection}' 追加 {len(documents)} 条文档")
            
        except Exception as e:
            # 检查是否是ID属性相关的错误
            if "'Document' object has no field 'id'" in str(e) or "'Document' object has no attribute 'id'" in str(e):
                logger.warning("检测到Document对象缺少ID属性，正在修复...")
                
                # 为文档添加唯一ID（通过创建新文档对象的方式）
                fixed_documents = []
                for doc in documents:
                    # 创建一个新的Document对象，确保它具有所有必需的属性
                    fixed_doc = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    )
                    fixed_documents.append(fixed_doc)
                
                # 重试添加文档
                try:
                    if self.vectorstore is None:
                        self.vectorstore = Chroma.from_documents(
                            documents=fixed_documents,
                            embedding=self.embeddings,
                            collection_name=target_collection,
                            persist_directory=self.persist_directory
                        )
                    else:
                        self.vectorstore.add_documents(fixed_documents)
                    
                    logger.info(f"成功修复并添加文档到集合 '{target_collection}'")
                except Exception as retry_e:
                    logger.error(f"重试添加文档失败: {retry_e}")
                    raise retry_e
            else:
                logger.error(f"添加文档到ChromaDB失败: {e}")
                raise e

    def search(self, query: str, k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 元数据过滤条件
            collection_name: 指定要搜索的集合名称（可选）
        
        Returns:
            (文档, 相似度分数) 列表
        """
        target_collection = collection_name or self.collection_name

        if self.vectorstore is None or (target_collection and self.collection_name != target_collection):
            try:
                # 检查集合是否存在
                client = chromadb.PersistentClient(path=self.persist_directory)
                collections = [c.name for c in client.list_collections()]
                
                if target_collection in collections:
                    self.vectorstore = Chroma(
                        collection_name=target_collection,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    self.collection_name = target_collection
                    logger.info(f"加载集合用于搜索: {target_collection}")
                else:
                    logger.warning("向量数据库未初始化")
                    return []
            except Exception as e:
                logger.error(f"加载ChromaDB集合失败: {e}")
                return []
        
        try:
            # ChromaDB 支持元数据过滤
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(query=query, k=k, filter=filter_dict)
            else:
                results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            
            logger.info(f"搜索查询: '{query}', 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取数据库信息
        """
        target_collection = collection_name or self.collection_name
        
        info = {
            "persist_directory": self.persist_directory,
            "collection_name": target_collection,
            "is_initialized": self.vectorstore is not None
        }
        
        try:
            # 检查集合是否存在
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [c.name for c in client.list_collections()]
            
            if target_collection in collections:
                # 获取集合信息
                collection = client.get_collection(name=target_collection)
                info["document_count"] = collection.count()
                
                # 如果 self.vectorstore 为空但集合存在，尝试初始化它
                if self.vectorstore is None:
                     try:
                        self.vectorstore = Chroma(
                            collection_name=target_collection,
                            embedding_function=self.embeddings,
                            persist_directory=self.persist_directory
                        )
                        info["is_initialized"] = True
                     except:
                        pass # 忽略加载错误，只返回统计
            else:
                info["document_count"] = 0
        except Exception as e:
            logger.error(f"获取ChromaDB集合信息失败: {e}")
            info["error"] = str(e)
        
        return info

    def clear_database(self):
        """清空ChromaDB集合"""
        try:
            # ChromaDB doesn't have a direct way to clear a collection, 
            # we need to delete and recreate it
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Delete the collection if it exists
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"ChromaDB集合 '{self.collection_name}' 已被删除")
            except ValueError:
                # Collection doesn't exist
                logger.info(f"ChromaDB集合 '{self.collection_name}' 不存在")
            
            # Clear the vectorstore reference
            self.vectorstore = None
        except Exception as e:
            logger.error(f"清空ChromaDB集合失败: {e}")