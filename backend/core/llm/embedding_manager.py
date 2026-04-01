"""
嵌入模型管理模块
专门处理嵌入模型的初始化和管理
"""

import os
import logging
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入模型管理器"""
    
    def __init__(self, 
                 embedding_model: str = None, 
                 dashscope_api_key: str = None):
        """
        初始化嵌入模型管理器
        
        Args:
            embedding_model: DashScope嵌入模型名称
            dashscope_api_key: DashScope API密钥
        """
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        
        # 初始化嵌入模型
        self.embeddings = self._init_embeddings()

    def _init_embeddings(self) -> Embeddings:
        """初始化嵌入模型"""
        try:
            # 确保 API Key 存在
            if not self.dashscope_api_key:
                logger.warning("未提供 DashScope API Key，将尝试从环境变量获取")
                self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

            embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key
            )
            # 简单测试 embedding 是否工作
            try:
                embeddings.embed_query("test")
                logger.info(f"成功加载并验证 DashScope嵌入模型: {self.embedding_model}")
                return embeddings
            except Exception as e:
                logger.error(f"DashScope 模型验证失败: {e}")
                raise e

        except Exception as e:
            logger.error(f"加载DashScope模型失败: {e}")
            logger.warning("使用备用HuggingFace模型")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成文档嵌入向量失败: {e}")
            return []

    def embed_query(self, query: str) -> List[float]:
        """为查询文本生成嵌入向量"""
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"生成查询嵌入向量失败: {e}")
            return []