# 项目安装与运行指南 (ChromaDB版)

本指南将帮助您使用 ChromaDB 替代 Milvus 来运行此 RAG 系统。

## 环境要求

- Python 3.8+
- Node.js (用于前端)

## 安装步骤

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd vector_databases
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

确保 [.env](file:///d:\work_ai\vector_databases\.env) 文件包含正确的配置：

```env
# ChromaDB Configuration
CHROMADB_PERSIST_DIR=./chroma_db
COLLECTION_NAME=agent_rag

# DashScope / LLM Configuration
DASHSCOPE_API_KEY=your_dashscope_api_key_here
EMBEDDING_MODEL=text-embedding-v3
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# Flask Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
```

### 4. 运行后端服务

```bash
python server.py
```

服务将在 `http://localhost:5000` 启动。

### 5. 运行前端服务

在另一个终端窗口中：

```bash
cd rag_front
npm install
npm run dev
```

前端将在 `http://localhost:5173` 启动。

## 测试 ChromaDB 功能

运行测试脚本来验证 ChromaDB 是否正常工作：

```bash
python test/chroma_test.py
```

## 上传文档到知识库

1. 使用 API 上传文档：
   ```bash
   curl -X POST http://localhost:5000/api/vector/upload_file \
     -F "file=@path/to/your/document.pdf" \
     -F "collection_name=your_collection_name"
   ```

2. 或使用后端提供的文档处理功能。

## 使用 API 查询

```bash
curl -X POST http://localhost:5000/api/vector/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "你的问题",
    "collection_name": "your_collection_name"
  }'
```

## 注意事项

- ChromaDB 是一个轻量级的向量数据库，适用于中小规模的数据集
- 数据将持久化在 `./chroma_db` 目录中
- 相比 Milvus，ChromaDB 更易于部署和维护，但在大规模数据场景下性能可能不如 Milvus