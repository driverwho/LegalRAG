"""
用途：将单个文档加载、切分、生成嵌入并上传到 Milvus 集合。
适用场景：命令行快速导入一个文件到指定集合，便于后续检索。
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFium2Loader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import connections, Collection, CollectionSchema, DataType, FieldSchema, utility

load_dotenv()

# --- 在这里定义您的配置参数 ---
FILE_PATH = "rag.txt"  # <-- 替换为您的文件路径
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agent_rag")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
# ----------------------------------------

class SimpleDocumentUploader:
    def __init__(self, host, port, collection_name, dashscope_api_key, embedding_model):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dashscope_api_key = dashscope_api_key
        self.embedding_model = embedding_model
        
        # 连接到Milvus
        self.connect_milvus()
        # 创建集合
        self.create_collection_if_not_exists()
        
    def connect_milvus(self):
        """连接到Milvus数据库"""
        connections.connect("default", host=self.host, port=self.port)
        print(f"已连接到 Milvus，地址为 {self.host}:{self.port}")
        
    def get_embedding(self, texts):
        """生成文本嵌入向量"""
        embeddings_model = DashScopeEmbeddings(
            model=self.embedding_model,
            dashscope_api_key=self.dashscope_api_key
        )
        return embeddings_model.embed_documents(texts)
        
    def get_schema(self):
        """定义Milvus集合的模式"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]
        return CollectionSchema(fields=fields, description="文本嵌入集合")
        
    def create_collection_if_not_exists(self):
        """创建集合（如果不存在）"""
        if not utility.has_collection(self.collection_name):
            schema = self.get_schema()
            collection = Collection(name=self.collection_name, schema=schema)
            
            # 创建索引
            index_params = {
                "index_type": "AUTOINDEX",
                "metric_type": "L2",
                "params": {}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            print(f"集合 '{self.collection_name}' 已创建并加载")
        else:
            print(f"集合 '{self.collection_name}' 已存在")
            
        self.collection = Collection(name=self.collection_name)
        
    def insert_data(self, names, texts, embeddings):
        """插入数据到Milvus"""
        # 准备数据，注意字段顺序要与schema定义一致（除了auto_id字段）
        # schema字段顺序：id(auto), name, text, embedding
        data = [
            names,                       # name字段  
            texts,                       # text字段
            embeddings                   # embedding字段
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        print(f"已向集合插入 {len(names)} 条记录")
        
    def process_file(self, file_path):
        """处理文件并上传到Milvus"""
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 {file_path}")
            return False
            
        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        extension = file_name.split(".")[-1].lower()
        
        # 根据文件类型选择加载器
        if extension == 'txt':
            loader = TextLoader(file_path, encoding='utf8')
        elif extension == 'pdf':
            loader = PyPDFium2Loader(file_path)
        elif extension == 'docx':
            loader = Docx2txtLoader(file_path)
        elif extension == 'csv':
            loader = CSVLoader(file_path)
        else:
            print(f"不支持的文件类型：{extension}")
            return False
            
        try:
            # 加载文档
            documents = loader.load()
            print(f"成功加载文档：{file_name}")
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            texts = [doc.page_content.strip() for doc in docs if doc.page_content and doc.page_content.strip()]
            final_texts = [t[:2000] for t in texts if t]
            
            # 生成嵌入向量
            print("正在生成嵌入向量...")
            embeddings = self.get_embedding(final_texts)
            
            # 准备文件名列表
            names = [file_name] * len(final_texts)
            
            # 插入数据
            self.insert_data(names, final_texts, embeddings)
            print(f"文档 '{file_name}' 上传成功！")
            return True
            
        except Exception as e:
            print(f"处理文件时出错：{e}")
            return False

def main():
    """主函数"""
    print("初始化文档上传器...")
    
    uploader = SimpleDocumentUploader(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )
    
    print(f"开始处理文档：{FILE_PATH}")
    success = uploader.process_file(FILE_PATH)
    
    if success:
        print("文档上传完成！")
        
        # 显示集合信息
        try:
            collection = Collection(name=COLLECTION_NAME)
            count = collection.num_entities
            print(f"集合 '{COLLECTION_NAME}' 中共有 {count} 条记录")
        except Exception as e:
            print(f"获取集合信息时出错：{e}")
    else:
        print("文档上传失败！")

if __name__ == "__main__":
    main()
