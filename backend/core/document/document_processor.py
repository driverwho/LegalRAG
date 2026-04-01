"""
文档处理器模块
专门处理各种格式文档的加载和预处理
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器类"""
    
    # 支持的文件类型
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.csv': 'csv',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.md': 'markdown'  # 添加对Markdown文件的支持
    }
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        初始化文档处理器
        
        Args:
            encoding: 文件编码格式
        """
        self.encoding = encoding

    def get_file_type(self, file_path: str) -> str:
        """
        获取文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件类型
        """
        extension = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(extension, 'unknown')

    def is_supported(self, file_path: str) -> bool:
        """
        检查文件是否支持
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否支持
        """
        return self.get_file_type(file_path) != 'unknown'

    def load_text_file(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        try:
            loader = TextLoader(file_path, encoding=self.encoding)
            documents = loader.load()
            
            # 添加文件信息到元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'text',
                    'file_name': Path(file_path).name
                })
            
            return documents
        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {e}")
            return []

    def load_csv_file(self, file_path: str, **kwargs) -> List[Document]:
        """加载CSV文件"""
        try:
            loader = CSVLoader(file_path, encoding=self.encoding, **kwargs)
            documents = loader.load()
            
            # 添加文件信息到元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'csv',
                    'file_name': Path(file_path).name
                })
            
            return documents
        except Exception as e:
            logger.error(f"加载CSV文件失败 {file_path}: {e}")
            return []

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """加载PDF文件"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 添加文件信息到元数据
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'pdf',
                    'file_name': Path(file_path).name,
                    'page_number': i + 1
                })
            
            return documents
        except Exception as e:
            logger.error(f"加载PDF文件失败 {file_path}: {e}")
            return []

    def load_docx_file(self, file_path: str) -> List[Document]:
        """加载Word文档"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # 添加文件信息到元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'docx',
                    'file_name': Path(file_path).name
                })
            
            return documents
        except Exception as e:
            logger.error(f"加载Word文档失败 {file_path}: {e}")
            return []

    def load_excel_file(self, file_path: str) -> List[Document]:
        """加载Excel文件"""
        try:
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()
            
            # 添加文件信息到元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'excel',
                    'file_name': Path(file_path).name
                })
            
            return documents
        except Exception as e:
            logger.error(f"加载Excel文件失败 {file_path}: {e}")
            return []

    def load_markdown_file(self, file_path: str) -> List[Document]:
        """加载Markdown文件"""
        try:
            # 尝试使用UnstructuredMarkdownLoader
            try:
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                loader = UnstructuredMarkdownLoader(file_path)
            except ImportError:
                # 如果没有安装unstructured库，回退到文本加载器
                logger.warning(f"未安装unstructured库，使用文本加载器处理Markdown文件: {file_path}")
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding=self.encoding)
                
            documents = loader.load()
            
            # 添加文件信息到元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'markdown',
                    'file_name': Path(file_path).name
                })
            
            return documents
        except Exception as e:
            logger.error(f"加载Markdown文件失败 {file_path}: {e}")
            return []

    def load_single_file(self, file_path: str, **kwargs) -> List[Document]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径
            **kwargs: 额外参数
            
        Returns:
            文档列表
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        file_type = self.get_file_type(file_path)
        
        if file_type == 'text':
            return self.load_text_file(file_path)
        elif file_type == 'csv':
            return self.load_csv_file(file_path, **kwargs)
        elif file_type == 'pdf':
            return self.load_pdf_file(file_path)
        elif file_type == 'docx':
            return self.load_docx_file(file_path)
        elif file_type == 'excel':
            return self.load_excel_file(file_path)
        elif file_type == 'markdown':
            return self.load_markdown_file(file_path)
        else:
            logger.warning(f"不支持的文件类型: {file_path}")
            # 尝试作为文本文件加载
            return self.load_text_file(file_path)

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        if not os.path.exists(file_path):
            return {"error": "文件不存在"}
        
        file_stat = os.stat(file_path)
        path_obj = Path(file_path)
        
        return {
            "file_name": path_obj.name,
            "file_path": str(path_obj.absolute()),
            "file_size": file_stat.st_size,
            "file_type": self.get_file_type(file_path),
            "is_supported": self.is_supported(file_path),
            "extension": path_obj.suffix.lower(),
            "created_time": file_stat.st_ctime,
            "modified_time": file_stat.st_mtime
        }