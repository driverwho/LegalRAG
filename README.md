# RAG 文档问答系统

一个企业级的检索增强生成（RAG）系统，支持多格式文档上传、智能文档预处理、向量存储和基于大语言模型的多轮对话问答。

## 功能介绍

### 核心功能

| 功能模块 | 描述 |
|---------|------|
| 📄 **多格式文档支持** | PDF、Word (DOCX/DOC)、TXT、CSV、Excel (XLSX/XLS)、Markdown |
| 🔍 **OCR 文字识别** | 基于 RapidOCR 的 PDF 和 Word 文档图片文字提取 |
| 🧹 **智能文档预处理** | 正则文本清洗 + LLM 法律文书智能纠错（`{}` 包裹输出 + 正则提取） |
| ✂️ **多策略文档分块** | 递归字符分块（通用）+ 父子分块（法律条文，按章/节/条层级） |
| 🧠 **向量存储** | 基于 ChromaDB 的持久化向量数据库 |
| 💬 **多轮对话 RAG** | 支持会话管理、对话历史持久化、上下文窗口自动压缩 |
| 🔄 **LLM 故障转移** | 可配置的多级 LLM 降级链（DashScope → Kimi → ...） |
| 📊 **质量检查** | 规则 + LLM 混合的文档质量评估 |
| ⚡ **异步处理** | Celery + Redis 异步文档处理管道，支持进度追踪和任务取消 |
| 📚 **知识库管理** | 集合/文档的完整 CRUD 操作，支持分页、关键词过滤 |

### 文档预处理管道

```
原始文档 → 正则清洗 → LLM 智能纠错 → 纯净文本
```

- **HTML/XML 标签去除** — 清理文档中的标记语言标签
- **特殊字符过滤** — 移除控制字符和乱码
- **日期格式标准化** — 统一转换为 `YYYY-MM-DD` 格式
- **标点符号规范化** — 半角转全角标点
- **敏感信息脱敏** — 身份证号、手机号自动替换为 `***`
- **LLM 智能纠错** — 法律文书专用校对（纠错、法律引用格式统一、法律简称补全）
- **LLM 输出净化** — 要求 LLM 将修改后的原文包裹在 `{}` 内，通过正则提取去除额外说明

### 文档分块策略

#### 通用分块 — RecursiveCharacterTextSplitter

面向普通文档，按字符长度递归分块，分隔符针对中英文混合内容优化：

```
分隔符优先级: \n\n → \n → 。 → ！ → ？ → ； → ， → 空格 → 逐字符
默认参数: chunk_size=500, chunk_overlap=50
```

#### 法律文档父子分块 — LegalParentChildSplitter

面向 Markdown 格式的法律条文，按 **章 → 节 → 条** 自然层级分割：

```
章 (Chapter)
├── 条 (Article)             ← 章无节时，章为父块
└── 节 (Section)
    └── 条 (Article)         ← 章有节时，节为父块
```

- **父块**：存储完整的章/节文本，用于上下文检索
- **子块**：存储单条法条文本，用于向量相似性搜索
- **元数据关联**：子块通过 `parent_chunk_id` 引用父块，并携带 `law_name`、`chapter`、`section`、`article_number` 等法条属性
- **中文数字解析**：内置中文数字转换（如 "二十三" → 23）

> 注：法律父子分块模块已实现，尚未接入主处理管道。使用方式见下方示例。

### 多轮对话与上下文管理

- **会话持久化** — 基于 SQLite + SQLAlchemy 的对话历史存储
- **Token 感知窗口管理** — 自动检测上下文是否超出 LLM 窗口限制
- **自动历史压缩** — 超出窗口时，保护最近 2 轮对话，压缩更早历史为摘要
- **会话 CRUD** — 创建、列表、查看、重命名、删除会话

### 元数据追踪

系统为每个文档和分块自动记录完整元数据：

```python
# 文档级元数据
doc_id, file_hash, file_size, uploaded_at, source, file_type

# 通用分块元数据
chunk_id, chunk_index, chunk_total, parent_doc_id, char_count

# 法律分块元数据 (父块)
chunk_type="parent", parent_chunk_id, law_name, chapter, section, child_count

# 法律分块元数据 (子块)
chunk_type="child", chunk_id, parent_chunk_id, law_name, chapter, section, article_number, article_title

# 处理元数据
preprocessed_by, preprocessed_at, preprocessor_version, processed_at
```

## 技术架构

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Vue.js    │◄────►│   FastAPI    │◄────►│  ChromaDB       │
│  Frontend   │      │   Backend    │      │  Vector Store   │
└─────────────┘      └──────┬───────┘      └─────────────────┘
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
             ┌──────────┐ ┌─────┐ ┌──────────────┐
             │  Celery  │ │SQLite│ │Context Manager│
             │  Worker  │ │ DB  │ │(Token-aware)  │
             └────┬─────┘ └─────┘ └──────────────┘
                  │
            ┌─────┼─────┐
            ▼           ▼
     ┌──────────┐ ┌──────────┐
     │  Redis   │ │ DashScope│
     │TaskQueue │ │ (LLM/Emb)│
     └──────────┘ └─────┬────┘
                        │ fallback
                  ┌─────▼────┐
                  │  Kimi    │
                  │ (备用LLM) │
                  └──────────┘
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
# ---- 必需 ----
# 阿里云 DashScope API (主 LLM 和嵌入模型)
DASHSCOPE_API_KEY=your_dashscope_api_key

# ---- 备用 LLM ----
# Moonshot Kimi API (降级备用)
MOONSHOT_API_KEY=your_moonshot_api_key

# 或使用 JSON 自定义降级链 (优先级高于 MOONSHOT_* 配置)
# LLM_FALLBACK_CHAIN=[{"name":"kimi","api_key_env":"MOONSHOT_API_KEY","base_url":"https://api.moonshot.cn/v1","model":"kimi-k2.5"}]

# ---- 可选配置（使用默认值可不设置）----
APP_HOST=0.0.0.0
APP_PORT=5000
COLLECTION_NAME=agent_rag
CHROMADB_PERSIST_DIR=./chroma_db
SQLITE_DB_PATH=./data/chat_history.db
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

所有接口统一前缀 `/api/vector`。

### 文档管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/upload_file` | 上传文件 (multipart/form-data，支持 OCR) |
| POST | `/upload_document` | 上传服务器本地文件 |

### 搜索与问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/query` | RAG 问答查询（支持 session_id 多轮对话） |
| POST | `/search` | 向量相似性搜索（无 LLM 生成） |

### 知识库管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/collections` | 列出所有集合及文档数 |
| GET | `/collection_info` | 获取指定集合的详细信息 |
| POST | `/clear_collection` | 清空指定集合 |
| GET | `/knowledge/{collection}/documents` | 分页查询文档列表（支持关键词过滤） |
| GET | `/knowledge/{collection}/documents/{id}` | 获取单个文档详情 |
| PUT | `/knowledge/{collection}/documents/{id}` | 更新文档内容或元数据 |
| DELETE | `/knowledge/{collection}/documents` | 批量删除文档 |

### 会话管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/sessions` | 创建新会话 |
| GET | `/sessions` | 列出所有会话 |
| GET | `/sessions/{session_id}` | 获取会话详情（含消息历史） |
| PUT | `/sessions/{session_id}` | 更新会话标题 |
| DELETE | `/sessions/{session_id}` | 删除会话 |

### 任务管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/tasks/{task_id}` | 查询异步任务状态和进度 |
| POST | `/tasks/{task_id}/cancel` | 取消正在执行的任务 |

### 示例请求

#### 上传文件

```bash
curl -X POST "http://localhost:5000/api/vector/upload_file" \
  -F "file=@/path/to/document.pdf" \
  -F "collection_name=agent_rag" \
  -F "use_ocr=false"
```

#### RAG 问答（多轮对话）

```bash
curl -X POST "http://localhost:5000/api/vector/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是善意取得？",
    "collection_name": "agent_rag",
    "session_id": "optional-session-uuid",
    "k": 5
  }'
```

#### 法律文档父子分块（代码调用）

```python
from backend.app.core.document.legal_splitter import LegalParentChildSplitter

splitter = LegalParentChildSplitter(law_name="中华人民共和国民法典")
parents, children = splitter.split(documents)

# parents  → 存储用于上下文检索
# children → 存入向量数据库用于语义搜索
```

## 项目结构

```
vector_databases/
├── backend/                        # FastAPI 后端
│   ├── app/
│   │   ├── main.py                # 应用入口
│   │   ├── config/
│   │   │   └── settings.py        # Pydantic 集中配置
│   │   ├── api/
│   │   │   ├── deps.py            # 依赖注入
│   │   │   ├── router.py          # 路由聚合
│   │   │   └── v1/
│   │   │       ├── documents.py   # 文档上传
│   │   │       ├── search.py      # 搜索/问答
│   │   │       ├── collections.py # 集合管理
│   │   │       ├── knowledge.py   # 知识库 CRUD
│   │   │       ├── sessions.py    # 会话管理
│   │   │       └── tasks.py       # 任务状态/取消
│   │   ├── core/
│   │   │   ├── document/          # 文档处理
│   │   │   │   ├── loader.py          # 多格式文档加载
│   │   │   │   ├── preprocessor.py    # 正则清洗 + LLM 纠错
│   │   │   │   ├── splitter.py        # 递归字符分块
│   │   │   │   ├── legal_splitter.py  # 法律文档父子分块 (*)
│   │   │   │   ├── pdf_loader.py      # RapidOCR PDF 加载
│   │   │   │   ├── docx_loader.py     # RapidOCR DOCX 加载
│   │   │   │   └── ocr.py            # OCR 引擎封装
│   │   │   ├── vector_store/      # 向量存储
│   │   │   │   ├── base.py            # 抽象接口
│   │   │   │   └── chroma.py          # ChromaDB 实现
│   │   │   ├── llm/               # LLM 管理
│   │   │   │   ├── chat.py            # 基础对话
│   │   │   │   ├── contextual_chat.py # 上下文感知对话
│   │   │   │   └── embedding.py       # 嵌入模型
│   │   │   ├── retriever/         # RAG 检索
│   │   │   │   ├── rag.py             # 基础 RAG 管道
│   │   │   │   ├── contextual_rag.py  # 多轮对话 RAG
│   │   │   │   └── contextual_integration.py
│   │   │   ├── context/           # 上下文窗口管理
│   │   │   │   └── context_manager.py # Token 感知压缩
│   │   │   ├── database/          # 数据库
│   │   │   │   ├── engine.py          # SQLAlchemy 引擎
│   │   │   │   ├── models.py          # ORM 模型
│   │   │   │   └── session_service.py # 会话/消息服务
│   │   │   ├── quality/           # 质量检查
│   │   │   │   ├── checker.py         # 规则 + LLM 混合检查
│   │   │   │   └── evaluate.py        # 质量评估
│   │   │   └── tasks/             # 异步任务
│   │   │       ├── document_tasks.py  # 文档处理任务
│   │   │       └── task_state.py      # 任务状态管理
│   │   ├── models/                # Pydantic 请求/响应模型
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── exceptions/            # 异常处理
│   │       └── handlers.py
│   └── celery_app.py              # Celery 配置
│
├── frontend/                       # Vue.js 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── KnowledgeBase.vue  # 知识库管理界面
│   │   │   └── RagChat.vue        # RAG 对话界面
│   │   ├── router/index.js        # 前端路由
│   │   └── main.js
│   └── package.json
│
├── docs/                           # 项目文档
│   ├── architecture.md            # 架构文档
│   └── context_management.md      # 上下文管理文档
│
├── tests/                          # 测试
│   └── unit/document/             # 文档处理单元测试
│
├── data/                           # SQLite 数据库目录
├── chroma_db/                      # ChromaDB 数据目录
└── requirements.txt                # Python 依赖
```

> (*) 标记模块已实现但尚未接入主管道。

## 配置说明

所有配置通过环境变量管理（`backend/.env`），由 Pydantic Settings 统一加载。

### 基础配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | - | 阿里云 DashScope API 密钥 (必需) |
| `MOONSHOT_API_KEY` | - | Moonshot API 密钥 (备用 LLM) |
| `LLM_MODEL` | qwen-plus | 主 LLM 模型 |
| `LLM_MODEL_MAX` | qwen3-max | 高阶 LLM 模型 |
| `EMBEDDING_MODEL` | text-embedding-v3 | 嵌入模型 |

### LLM 降级链

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_FALLBACK_CHAIN` | - | JSON 数组，自定义降级提供商链 |
| `LLM_FALLBACK_MAX_RETRIES` | 3 | 每个提供商最大重试次数 |
| `LLM_FALLBACK_RETRY_DELAY` | 2.0 | 重试基础延迟（秒，指数退避） |

### 文档处理

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CHUNK_SIZE` | 500 | 文档分块大小 |
| `CHUNK_OVERLAP` | 50 | 分块重叠大小 |
| `ENABLE_PREPROCESSING` | true | 启用文档预处理管道 |
| `ENABLE_LLM_PREPROCESSING` | true | 启用 LLM 纠错阶段 |
| `ENABLE_QUALITY_CHECK` | true | 启用文档质量检查 |

### 上下文窗口

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CONTEXT_WINDOW_SIZE` | 30000 | LLM 上下文窗口大小 (tokens) |
| `CONTEXT_RESERVED_OUTPUT_TOKENS` | 2000 | 预留给模型输出的 token 数 |
| `CONTEXT_PROTECTED_ROUNDS` | 2 | 压缩时保护的最近对话轮数 |

### 存储

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CHROMADB_PERSIST_DIR` | ./chroma_db | ChromaDB 持久化目录 |
| `SQLITE_DB_PATH` | ./data/chat_history.db | 会话历史数据库路径 |
| `COLLECTION_NAME` | agent_rag | 默认集合名称 |

### 检索

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SIMILARITY_THRESHOLD` | 0.5 | 最低相似度阈值 |
| `MAX_RESULTS` | 10 | 最大返回结果数 |

## 开发指南

### 运行测试

```bash
pytest tests/
```

### 代码检查

```bash
# 格式化代码
black backend/app/

# 类型检查
mypy backend/app/
```

### 添加新的文档格式

在 `backend/app/core/document/loader.py` 中：
1. 向 `SUPPORTED_EXTENSIONS` 添加扩展名映射
2. 实现对应的 `_load_{format}` 方法
3. 在 `__init__` 的 `_loaders` 字典中注册

### 添加新的 LLM 提供商

在 `.env` 中通过 `LLM_FALLBACK_CHAIN` 配置 JSON 数组即可，无需修改代码：

```env
LLM_FALLBACK_CHAIN=[{"name":"kimi","api_key_env":"MOONSHOT_API_KEY","base_url":"https://api.moonshot.cn/v1","model":"kimi-k2.5"},{"name":"deepseek","api_key_env":"DEEPSEEK_API_KEY","base_url":"https://api.deepseek.com/v1","model":"deepseek-chat"}]
```

## 常见问题

### Q: Windows 上 Celery 报错？
A: Windows 需要使用 `solo` pool，启动命令中已包含 `--pool=solo`。

### Q: LLM 预处理失败怎么办？
A: 系统会自动按降级链逐一尝试备用 LLM。所有提供商均失败后，文档会标记 `pending_preprocessing=True`，仅保留正则清洗结果，可稍后重试。

### Q: 如何使用法律文档父子分块？
A: 该模块已实现但尚未接入主管道，需直接导入使用：

```python
from backend.app.core.document.legal_splitter import LegalParentChildSplitter

splitter = LegalParentChildSplitter(law_name="中华人民共和国民法典")
parents, children = splitter.split(documents)
```

### Q: 对话历史存储在哪里？
A: 存储在 SQLite 数据库中，默认路径 `./data/chat_history.db`，通过 `SQLITE_DB_PATH` 环境变量配置。

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 PR！请确保：
1. 代码通过所有测试
2. 遵循 PEP 8 代码规范
3. 更新相关文档

---

**版本**: 3.0.0  
**技术栈**: FastAPI + Vue.js + ChromaDB + SQLite + Celery + Redis + DashScope
