"""
ChromaDB 测试工具
功能：
1. 创建和管理ChromaDB集合
2. 向集合中添加文档
3. 执行相似性搜索
4. 检查集合状态和内容
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
import sys
sys.path.insert(0, sys.path[0]+"/../")
from backend.core.vector_db.chromadb_manager import VectorDatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chroma_setup():
    """测试ChromaDB的基本设置和功能"""
    print("🧪 开始测试 ChromaDB 设置...")
    
    try:
        # 初始化数据库管理器
        manager = VectorDatabaseManager(
            collection_name="test_collection",
            persist_directory="./chroma_test_db"
        )
        print("✅ ChromaDB 连接成功!")
        
        # 测试数据库信息
        info = manager.get_database_info()
        print(f"📊 数据库信息: {info}")
        
        return manager
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None

def test_add_documents(manager):
    """测试添加文档功能"""
    print("\n📝 测试添加文档...")
    
    try:
        # 创建测试文档
        from langchain_core.documents import Document
        
        test_docs = [
            Document(page_content="人工智能是计算机科学的一个分支", metadata={"source": "ai_intro"}),
            Document(page_content="机器学习是人工智能的一个子领域", metadata={"source": "ml_intro"}),
            Document(page_content="深度学习使用神经网络进行学习", metadata={"source": "dl_intro"}),
            Document(page_content="自然语言处理帮助计算机理解人类语言", metadata={"source": "nlp_intro"}),
            Document(page_content="计算机视觉使计算机能够识别图像内容", metadata={"source": "cv_intro"})
        ]
        
        # 添加到数据库
        manager.add_documents_to_db(test_docs)
        print("✅ 文档添加成功!")
        
        # 检查数据库状态
        info = manager.get_database_info()
        print(f"📊 更新后的数据库信息: {info}")
        
    except Exception as e:
        print(f"❌ 添加文档失败: {e}")

def test_search(manager):
    """测试搜索功能"""
    print("\n🔍 测试搜索功能...")
    
    try:
        # 执行搜索
        results = manager.search(query="什么是人工智能", k=3)
        
        print(f"找到 {len(results)} 个结果:")
        for i, (doc, score) in enumerate(results):
            print(f"  [{i+1}] 相似度: {score:.4f}")
            print(f"      内容: {doc.page_content}")
            print(f"      来源: {doc.metadata}")
            print()
        
    except Exception as e:
        print(f"❌ 搜索失败: {e}")

def test_collection_management():
    """测试集合管理功能"""
    print("\n🗂️ 测试集合管理...")
    
    try:
        manager = VectorDatabaseManager(
            collection_name="temp_collection",
            persist_directory="./chroma_test_db"
        )
        
        # 添加一些文档
        from langchain_core.documents import Document
        temp_docs = [
            Document(page_content="这是一个临时测试文档", metadata={"source": "temp_test"})
        ]
        manager.add_documents_to_db(temp_docs)
        
        # 检查信息
        info = manager.get_database_info()
        print(f"📊 临时集合信息: {info}")
        
        # 清空集合
        manager.clear_database()
        print("🗑️ 集合已清空")
        
        # 再次检查信息
        info_after_clear = manager.get_database_info()
        print(f"📊 清空后的信息: {info_after_clear}")
        
    except Exception as e:
        print(f"❌ 集合管理测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始 ChromaDB 功能测试\n")
    
    # 测试基本设置
    manager = test_chroma_setup()
    if not manager:
        return
    
    # 测试添加文档
    test_add_documents(manager)
    
    # 测试搜索功能
    test_search(manager)
    
    # 测试集合管理
    test_collection_management()
    
    print("\n🎉 ChromaDB 测试完成!")

if __name__ == "__main__":
    main()