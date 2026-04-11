# RAG 文档问答系统

一个企业级的检索增强生成（RAG）系统，支持多格式文档上传、智能文档预处理、向量存储和基于大语言模型的问答。

## 功能介绍

### 核心功能

| 功能模块 | 描述 |
|---------|------|
| 📄 **多格式文档支持** | PDF、Word (DOCX/DOC)、TXT、CSV、Excel (XLSX/XLS)、Markdown |
| 🔍 **OCR 文字识别** | 支持 PDF 和 Word 文档中的图片文字提取 |
| 🧹 **智能文档预处理** | 基于正则的文本清洗（去 HTML 标签、特殊字符、敏感信息脱敏）+ LLM 智能纠错 |
| ✂️ **智能文档分块** | 递归字符分块，针对中英文优化分隔符 |
| 🧠 **向量存储** | 基于 ChromaDB 的持久化向量数据库 |
| 💬 **RAG 问答** | 相似性检索 + LLM 生成，支持置信度评分 |
| 🔄 **LLM 故障转移** | 主 LLM 失败时自动降级到备用提供商 |
| 📊 **质量检查** | 文档处理前后的质量对比分析 |
| ⚡ **异步处理** | Celery + Redis 异步文档处理管道 |

### 文档预处理能力

- **HTML/XML 标签去除** - 清理文档中的标记语言标签
- **特殊字符过滤** - 移除控制字符和乱码
- **日期格式标准化** - 统一转换为 `YYYY-MM-DD` 格式
- **标点符号规范化** - 半角转全角标点
- **敏感信息脱敏** - 身份证号、手机号自动替换为 `***`
- **LLM 智能纠错** - 法律文书专用纠错模型

### 元数据追踪

系统为每个文档和分块自动记录完整元数据：

```python
# 文档级元数据
doc_id, file_hash, file_size, uploaded_at, source, file_type

# 分块级元数据  
chunk_id, chunk_index, chunk_total, parent_doc_id, char_count

# 处理元数据
preprocessed_by, preprocessed_at, preprocessor_version, processed_at
```

## 技术架构

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Vue.js    │◄────►│   FastAPI    │◄────►│  ChromaDB       │
│  Frontend   │      │   Backend    │      │  Vector Store   │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐      ┌─────────────────┐
                     │    Celery    │◄────►│     Redis       │
                     │   Worker     │      │  Task Queue     │
                     └──────────────┘      └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   DashScope  │
                     │   (LLM/Emb)  │
                     └──────────────┘
```

## 环境要求

- **Python**: 3.10+
- **Node.js**: 20.19.0+ 或 >=22.12.0
- **Redis**: 5.0+ (Celery 消息队列)
- **操作系统**: Windows / Linux / macOS

## 快速启动

### 1. 克隆项目并进入目录

```bash
cd vector_databases
```

### 2. 后端启动

#### 2.1 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

#### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

#### 2.3 配置环境变量

在 `backend/` 目录下创建 `.env` 文件：

```env
# 阿里云 DashScope API (主 LLM 和嵌入模型)
DASHSCOPE_API_KEY=your_dashscope_api_key

# Moonshot Kimi API (备用 LLM)
MOONSHOT_API_KEY=your_moonshot_api_key

# 可选配置（使用默认值可不设置）
APP_HOST=0.0.0.0
APP_PORT=5000
COLLECTION_NAME=agent_rag
CHROMADB_PERSIST_DIR=./chroma_db
UPLOAD_FOLDER=./temp/vector_uploads
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

#### 2.4 启动 Redis

```bash
# Windows (使用 WSL 或 Docker)
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Linux/macOS
redis-server
```

#### 2.5 启动 Celery Worker
目前仅支持单进程。
```bash
cd backend
celery -A backend.celery_app worker --loglevel=info --pool=solo  
```

#### 2.6 启动 FastAPI 服务

```bash
# 方式 1: 直接运行
python -m app.main

# 方式 2: 使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

服务启动后访问:
- API 文档: http://localhost:5000/docs
- 备用文档: http://localhost:5000/redoc

### 3. 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端默认运行在 http://localhost:5173

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 健康检查 |
| POST | `/api/vector/upload_file` | 上传文件 (multipart/form-data) |
| POST | `/api/vector/upload_document` | 上传服务器本地文件 |
| POST | `/api/vector/query` | RAG 问答查询 |
| POST | `/api/vector/search` | 向量相似性搜索 |
| GET | `/api/vector/collection_info` | 获取集合信息 |
| POST | `/api/vector/clear_collection` | 清空集合 |

### 示例请求

#### 上传文件

```bash
curl -X POST "http://localhost:5000/api/vector/upload_file" \
  -F "file=@/path/to/document.pdf" \
  -F "collection_name=agent_rag"
```

#### RAG 问答

```bash
curl -X POST "http://localhost:5000/api/vector/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是善意取得？",
    "collection_name": "agent_rag",
    "k": 5
  }'
```

## 项目结构

```
vector_databases/
├── backend/                    # FastAPI 后端
│   ├── app/
│   │   ├── main.py            # 应用入口
│   │   ├── config/            # 配置管理
│   │   ├── api/               # API 路由
│   │   │   ├── v1/
│   │   │   │   ├── documents.py   # 文档上传
│   │   │   │   ├── search.py      # 搜索/问答
│   │   │   │   └── collections.py # 集合管理
│   │   ├── core/              # 核心业务逻辑
│   │   │   ├── document/      # 文档处理
│   │   │   │   ├── loader.py      # 文档加载
│   │   │   │   ├── splitter.py    # 文档分块
│   │   │   │   └── preprocessor.py # 文档预处理
│   │   │   ├── vector_store/  # 向量存储
│   │   │   ├── llm/           # LLM 管理
│   │   │   └── retriever/     # RAG 检索
│   │   └── tasks/             # Celery 异步任务
│   ├── celery_app.py          # Celery 配置
│   └── requirements.txt       # Python 依赖
│
├── frontend/                   # Vue.js 前端
│   ├── src/
│   │   └── components/        # Vue 组件
│   └── package.json
│
├── docs/                       # 项目文档
│   └── architecture.md        # 架构文档
│
└── chroma_db/                  # ChromaDB 数据目录
```

## 配置说明

所有配置通过环境变量管理，主要配置项：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | - | 阿里云 DashScope API 密钥 (必需) |
| `MOONSHOT_API_KEY` | - | Moonshot API 密钥 (备用 LLM) |
| `EMBEDDING_MODEL` | text-embedding-v3 | 嵌入模型 |
| `LLM_MODEL` | qwen-plus | 主 LLM 模型 |
| `CHUNK_SIZE` | 500 | 文档分块大小 |
| `CHUNK_OVERLAP` | 50 | 分块重叠大小 |
| `SIMILARITY_THRESHOLD` | 0.5 | 相似度阈值 |

## 开发指南

### 运行测试

```bash
cd backend
pytest
```

### 代码检查

```bash
# 格式化代码
black app/

# 类型检查
mypy app/
```

## 常见问题

### Q: Windows 上 Celery 报错？
A: Windows 需要使用 `solo` pool，已自动配置：
```python
celery_app.conf.worker_pool = "solo"
```

### Q: 如何添加新的文档格式支持？
A: 在 `backend/app/core/document/loader.py` 中添加：
1. 扩展名到 `SUPPORTED_EXTENSIONS`
2. 实现对应的 `_load_{format}` 方法

### Q: LLM 预处理失败怎么办？
A: 系统会自动降级到备用 LLM (如 Kimi)，失败文档会标记 `pending_preprocessing=True`，可稍后重试。

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 PR！请确保：
1. 代码通过所有测试
2. 遵循 PEP 8 代码规范
3. 更新相关文档

---

**版本**: 2.0.0  
**技术栈**: FastAPI + Vue.js + ChromaDB + Celery + DashScope
