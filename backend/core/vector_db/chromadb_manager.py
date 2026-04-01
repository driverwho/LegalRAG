"""
向量存储模块
负责 ChromaDB 的具体存储和检索操作
"""

import os
import logging
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class VectorStorage:
    """向量存储实现类"""
    
    def __init__(self, 
                 collection_name: str,
                 embeddings: Embeddings,
                 persist_directory: str = "./chroma_db"):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            embeddings: 嵌入模型
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        # 确保持久化目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
    def _get_or_create_vectorstore(self, collection_name: str = None):
        """获取或创建 vectorstore 实例"""
        target_collection = collection_name or self.collection_name
        
        if self.vectorstore is not None and self.collection_name == target_collection:
            return self.vectorstore
            
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [c.name for c in client.list_collections()]
            
            if target_collection in collections:
                self.vectorstore = Chroma(
                    collection_name=target_collection,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"加载现有集合: {target_collection}")
            else:
                # 集合不存在时不立即创建，等待添加文档时创建
                logger.info(f"集合 {target_collection} 不存在，将在添加文档时创建")
                return None
                
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"获取/创建 vectorstore 失败: {e}")
            return None
    
    def add_documents(self, documents: List[Document], collection_name: str = None):
        """添加文档到向量数据库"""
        if not documents:
            logger.warning("没有文档需要添加")
            return
        
        target_collection = collection_name or self.collection_name
        
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [c.name for c in client.list_collections()]
            
            if target_collection in collections:
                # 集合存在，加载并追加
                vectorstore = Chroma(
                    collection_name=target_collection,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                vectorstore.add_documents(documents)
                logger.info(f"向现有集合 '{target_collection}' 追加 {len(documents)} 条文档")
            else:
                # 集合不存在，创建并插入
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=target_collection,
                    persist_directory=self.persist_directory
                )
                logger.info(f"创建集合 '{target_collection}' 并插入 {len(documents)} 条文档")
            
            self.vectorstore = vectorstore
            self.collection_name = target_collection
            
        except Exception as e:
            # 处理 Document ID 相关问题
            if "'Document' object has no field 'id'" in str(e) or "'Document' object has no attribute 'id'" in str(e):
                logger.warning("检测到 Document 对象缺少 ID 属性，正在修复...")
                
                fixed_documents = []
                for doc in documents:
                    fixed_doc = Document(
                        page_content=doc.page_content,
                        metadata=doc.metadata
                    )
                    fixed_documents.append(fixed_doc)
                
                try:
                    if target_collection in collections:
                        vectorstore = Chroma(
                            collection_name=target_collection,
                            embedding_function=self.embeddings,
                            persist_directory=self.persist_directory
                        )
                        vectorstore.add_documents(fixed_documents)
                    else:
                        vectorstore = Chroma.from_documents(
                            documents=fixed_documents,
                            embedding=self.embeddings,
                            collection_name=target_collection,
                            persist_directory=self.persist_directory
                        )
                    
                    self.vectorstore = vectorstore
                    self.collection_name = target_collection
                    logger.info(f"成功修复并添加文档到集合 '{target_collection}'")
                except Exception as retry_e:
                    logger.error(f"重试添加文档失败: {retry_e}")
                    raise retry_e
            else:
                logger.error(f"添加文档失败: {e}")
                raise e
    
    def search(self, query: str, k: int = 5, filter_dict: Dict = None, 
               collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """相似性搜索"""
        target_collection = collection_name or self.collection_name
        
        vectorstore = self._get_or_create_vectorstore(target_collection)
        if vectorstore is None:
            # 尝试重新加载
            try:
                client = chromadb.PersistentClient(path=self.persist_directory)
                collections = [c.name for c in client.list_collections()]
                
                if target_collection in collections:
                    vectorstore = Chroma(
                        collection_name=target_collection,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    self.vectorstore = vectorstore
                    logger.info(f"加载集合用于搜索: {target_collection}")
                else:
                    logger.warning(f"集合 {target_collection} 不存在")
                    return []
            except Exception as e:
                logger.error(f"加载集合失败: {e}")
                return []
        
        try:
            if filter_dict:
                results = vectorstore.similarity_search_with_score(
                    query=query, k=k, filter=filter_dict
                )
            else:
                results = vectorstore.similarity_search_with_score(query=query, k=k)
            
            logger.info(f"搜索查询: '{query}', 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """获取数据库信息"""
        target_collection = collection_name or self.collection_name
        
        info = {
            "persist_directory": self.persist_directory,
            "collection_name": target_collection,
            "is_initialized": self.vectorstore is not None
        }
        
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [c.name for c in client.list_collections()]
            
            if target_collection in collections:
                collection = client.get_collection(name=target_collection)
                info["document_count"] = collection.count()
                
                if self.vectorstore is None:
                    try:
                        self.vectorstore = Chroma(
                            collection_name=target_collection,
                            embedding_function=self.embeddings,
                            persist_directory=self.persist_directory
                        )
                        info["is_initialized"] = True
                    except:
                        pass
            else:
                info["document_count"] = 0
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            info["error"] = str(e)
        
        return info
    
    def clear_database(self):
        """清空集合"""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"集合 '{self.collection_name}' 已被删除")
            except ValueError:
                logger.info(f"集合 '{self.collection_name}' 不存在")
            
            self.vectorstore = None
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
"""
向量数据库管理模块
基于LangChain和ChromaDB实现文档切分、向量化存储和检索功能
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .vector_storage import VectorStorage
from ..document.document_processor import DocumentProcessor
from ..llm.embedding_manager import EmbeddingManager
from langchain_community.embeddings import DashScopeEmbeddings


class VectorDatabaseManager:
    """向量数据库管理器 (ChromaDB后端)"""
    
    def __init__(self, 
                 collection_name: str = None,
                 embedding_model: str = None,
                 dashscope_api_key: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 persist_directory: str = "./chroma_db"):  # 添加持久化目录
        """
        初始化向量数据库管理器
        
        Args:
            collection_name: ChromaDB 集合名称
            embedding_model: DashScope嵌入模型名称
            dashscope_api_key: DashScope API密钥
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
            persist_directory: ChromaDB持久化目录
        """
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        
        # 初始化嵌入模型管理器
        self.embedding_manager = EmbeddingManager(
            embedding_model=self.embedding_model,
            dashscope_api_key=self.dashscope_api_key
        )
        self.embeddings = self.embedding_manager.embeddings
        
        # 初始化文档处理器
        self.doc_processor = DocumentProcessor(encoding='utf-8')
        
        # 初始化文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 初始化向量存储
        self.vector_storage = VectorStorage(
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # 向量数据库实例
        self.vectorstore = self.vector_storage.vectorstore

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 确保 API Key 存在
            if not self.dashscope_api_key:
                 logger.warning("未提供 DashScope API Key，将尝试从环境变量获取")
                 self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key
            )
            # 简单测试 embedding 是否工作
            try:
                self.embeddings.embed_query("test")
                logger.info(f"成功加载并验证 DashScope嵌入模型: {self.embedding_model}")
            except Exception as e:
                 logger.error(f"DashScope 模型验证失败: {e}")
                 raise e

        except Exception as e:
            logger.error(f"加载DashScope模型失败: {e}")
            logger.warning("使用备用HuggingFace模型")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            documents = self.doc_processor.load_single_file(file_path)
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            return documents

    def process_file(self, file_path: str, collection_name: str = None) -> bool:
        """
        处理单个文件：加载、切分、存储
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            
        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            documents = self.load_document(file_path)
            if not documents:
                return False
            
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs, collection_name)
            
            logger.info(f"文件处理完成: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def process_csv_data(self, csv_path: str,
                         text_columns: List[str] = None,
                         metadata_columns: List[str] = None) -> bool:
        """
        处理CSV数据文件
        
        Args:
            csv_path: CSV文件路径
            text_columns: 需要向量化的文本列名列表
            
        Returns:
            处理是否成功
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
            if text_columns is None or not text_columns:
                object_cols = [c for c in df.columns if df[c].dtype == 'object']
                text_columns = [c for c in object_cols if not str(c).startswith('Unnamed')]

            if metadata_columns is None:
                metadata_columns = [c for c in df.columns if c not in text_columns]

            documents = []
            for idx, row in df.iterrows():
                content_parts = []
                metadata = {"source": csv_path, "row_index": idx}
                
                for col in text_columns:
                    if pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            content_parts.append(f"{col}: {text}")

                for col in metadata_columns:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[str(col)] = str(val)
                
                if content_parts:
                    content = "\n".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"构建了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs)
            
            return True
            
        except Exception as e:
            logger.error(f"处理CSV数据失败: {e}")
            return False

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """使用嵌入模型为一组文本生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []

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
        return self.vector_storage.search(
            query=query, 
            k=k, 
            filter_dict=filter_dict, 
            collection_name=collection_name
        )

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取数据库信息
        """
        return self.vector_storage.get_database_info(collection_name)

    def clear_database(self):
        """清空ChromaDB集合"""
        self.vector_storage.clear_database()

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
        """
        将文档添加到ChromaDB数据库
        
        Args:
            documents: 文档列表
            collection_name: 集合名称（可选，覆盖默认）
        """
        self.vector_storage.add_documents(documents, collection_name)


# 全局连接变量
db_manager = None
initialized = False  # 添加初始化标志

def init_db_manager():
    """初始化数据库管理器"""
    global db_manager, initialized
    
    if initialized:
        # 如果已经初始化，直接返回现有的实例
        return db_manager
    
    try:
        # 从环境变量获取配置
        collection_name = os.getenv("COLLECTION_NAME", "agent_rag")
        persist_directory = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_db")
        
        db_manager = VectorDatabaseManager(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        initialized = True
        logger.info("VectorDatabaseManager 初始化完成")
    except Exception as e:
        logger.error(f"初始化 VectorDatabaseManager 失败: {e}")
        raise
    return db_manager

def get_db_manager():
    """获取数据库管理器实例"""
    global db_manager
    if db_manager is None:
        init_db_manager()
    return db_manager
