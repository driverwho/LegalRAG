"""
向量数据库管理模块
基于LangChain和Milvus实现文档切分、向量化存储和检索功能
- 1.
MinIO (对象存储服务) : Milvus 使用 MinIO 来存储数据。它提供了一个网页管理界面。

- 登录页面 : http://localhost:9001
- 账号 (Access Key) : minioadmin
- 密码 (Secret Key) : minioadmin
- 2.
Milvus (向量数据库) : 在我们当前的配置中，Milvus 本身 没有 提供一个用于登录的网页界面。您可以通过代码（例如使用 pymilvus 库）连接到 Milvus 服务来进行操作。

- 连接地址 : localhost:19530
"""

import os
import logging
import uuid
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
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
)
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb

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
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 初始化ChromaDB客户端
        self._init_chroma_client()
        
        # 向量数据库实例
        self.vectorstore = None
        # 延迟加载：不要在 __init__ 中调用 _load_existing_db，避免启动时连接未就绪或报错
        # self._load_existing_db() 

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

    def _init_chroma_client(self):
        """初始化ChromaDB客户端"""
        try:
            # 创建持久化目录
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"ChromaDB持久化目录: {self.persist_directory}")
        except Exception as e:
            logger.error(f"初始化ChromaDB目录失败: {e}")
            raise

    def _load_existing_db(self):
        """加载已存在的ChromaDB集合"""
        try:
            # 检查集合是否存在
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [c.name for c in client.list_collections()]
            
            if self.collection_name in collections:
                try:
                    self.vectorstore = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                    logger.info(f"成功加载现有ChromaDB集合: {self.collection_name}")
                except Exception as e:
                    logger.error(f"加载现有集合失败: {e}")
                    self.vectorstore = None
            else:
                logger.info(f"未找到集合 {self.collection_name}，将在添加文档时创建")
        except Exception as e:
            logger.error(f"加载ChromaDB集合失败: {e}")

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == '.md':  # 添加对Markdown文件的支持
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                logger.warning(f"未识别的文件类型 {file_extension}, 使用文本加载器")
            
            documents = loader.load()
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

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
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
                import uuid
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
