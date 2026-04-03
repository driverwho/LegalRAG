"""
ChromaDB 测试工具
功能：
1. 创建和管理ChromaDB集合
2. 向集合中添加文档
3. 执行相似性搜索
4. 检查集合状态和内容
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.core.vector_store.chroma import ChromaVectorStore

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_embeddings():
    """获取 embedding 模型（兼容未配置 API Key 的环境）。"""
    try:
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(model="text-embedding-v3")
    except Exception:
        from langchain_community.embeddings import FakeEmbeddings

        logger.warning("DashScope 不可用，使用 FakeEmbeddings 进行测试")
        return FakeEmbeddings(size=1536)


def test_chroma_setup():
    """测试ChromaDB的基本设置和功能"""
    print("🧪 开始测试 ChromaDB 设置...")

    try:
        embeddings = _get_embeddings()
        store = ChromaVectorStore(
            collection_name="test_collection",
            embeddings=embeddings,
            persist_directory="./chroma_test_db",
        )
        print("✅ ChromaDB 连接成功!")

        info = store.get_collection_info()
        print(f"📊 数据库信息: {info}")

        return store
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None


def test_add_documents(store):
    """测试添加文档功能"""
    print("\n📝 测试添加文档...")

    try:
        from langchain_core.documents import Document

        test_docs = [
            Document(
                page_content="人工智能是计算机科学的一个分支",
                metadata={"source": "ai_intro"},
            ),
            Document(
                page_content="机器学习是人工智能的一个子领域",
                metadata={"source": "ml_intro"},
            ),
            Document(
                page_content="深度学习使用神经网络进行学习",
                metadata={"source": "dl_intro"},
            ),
            Document(
                page_content="自然语言处理帮助计算机理解人类语言",
                metadata={"source": "nlp_intro"},
            ),
            Document(
                page_content="计算机视觉使计算机能够识别图像内容",
                metadata={"source": "cv_intro"},
            ),
        ]

        store.add_documents(test_docs)
        print("✅ 文档添加成功!")

        info = store.get_collection_info()
        print(f"📊 更新后的数据库信息: {info}")

    except Exception as e:
        print(f"❌ 添加文档失败: {e}")


def test_search(store):
    """测试搜索功能"""
    print("\n🔍 测试搜索功能...")

    try:
        results = store.search(query="什么是人工智能", k=3)

        print(f"找到 {len(results)} 个结果:")
        for i, (doc, score) in enumerate(results):
            print(f"  [{i + 1}] 相似度: {score:.4f}")
            print(f"      内容: {doc.page_content}")
            print(f"      来源: {doc.metadata}")
            print()

    except Exception as e:
        print(f"❌ 搜索失败: {e}")


def test_collection_management():
    """测试集合管理功能"""
    print("\n🗂️ 测试集合管理...")

    try:
        embeddings = _get_embeddings()
        store = ChromaVectorStore(
            collection_name="temp_collection",
            embeddings=embeddings,
            persist_directory="./chroma_test_db",
        )

        from langchain_core.documents import Document

        temp_docs = [
            Document(
                page_content="这是一个临时测试文档", metadata={"source": "temp_test"}
            )
        ]
        store.add_documents(temp_docs)

        info = store.get_collection_info()
        print(f"📊 临时集合信息: {info}")

        store.clear_collection()
        print("🗑️ 集合已清空")

        info_after_clear = store.get_collection_info()
        print(f"📊 清空后的信息: {info_after_clear}")

    except Exception as e:
        print(f"❌ 集合管理测试失败: {e}")


def main():
    """主测试函数"""
    print("🚀 开始 ChromaDB 功能测试\n")

    store = test_chroma_setup()
    if not store:
        return

    test_add_documents(store)
    test_search(store)
    test_collection_management()

    print("\n🎉 ChromaDB 测试完成!")


if __name__ == "__main__":
    main()
