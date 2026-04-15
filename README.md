# LegalRAG - 法律智能问答系统

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.5+-4FC08D.svg?logo=vue.js&logoColor=white)](https://vuejs.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-FF6F00.svg)](https://www.trychroma.com)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.0+-2B6CB0.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个面向法律领域的** Agent 驱动 RAG (Retrieval-Augmented Generation) 系统**，支持意图识别、混合检索、流式输出和多级 LLM 故障转移。

> **核心定位**：通过意图路由和混合检索，实现法律文档的智能问答与精准引用。

## 🚀 核心特性

### 1. Agent 驱动的智能路由 (v2)

系统基于 `LegalRouterAgent` 实现意图驱动的查询路由：

```
用户提问
    ↓
查询预处理 (分类/纠错/元数据提取)
    ↓
意图识别 → 工具选择 (法条检索/案例检索/通用问答)
    ↓
并行检索 → 结果融合 → LLM生成
    ↓
流式输出 + 引用溯源
```

- **查询分类器**：自动识别问题类型（法条查询/案例分析/通用问题）
- **智能纠错**：LLM 辅助的法律术语拼写纠正
- **元数据提取**：自动提取时间、地区、法院等过滤条件
- **动态工具路由**：根据问题复杂度选择检索策略

### 2. 混合检索引擎

**向量检索 + BM25 关键词检索 + RRF 融合**

| 检索方式 | 技术实现 | 适用场景 |
|---------|---------|---------|
| **向量检索** | ChromaDB + DashScope Embedding | 语义相似、概念匹配 |
| **关键词检索** | BM25Okapi + jieba 中文分词 | 精确匹配、法条编号 |
| **融合策略** | RRF (Reciprocal Rank Fusion) | 综合排序、提升召回 |

- 并行执行向量检索和 BM25 检索
- RRF 算法融合两种结果，k=60 平滑常数
- 支持元数据过滤（时间、地区、文档类型）

### 3. 异步 RAG 流水线

**5 阶段异步处理架构**：

```python
Stage 1: 查询预处理 (preprocess_query)
         └─ 分类/纠错/元数据提取

Stage 2: 混合检索 (search_hybrid_async)
         ├─ 向量检索 (async)
         ├─ BM25 检索 (async)
         └─ RRF 融合

Stage 3: 上下文组装 (assemble_context_async)
         ├─ 法条查询 → 父块扩展
         └─ 案例查询 → 滑动窗口

Stage 4: 重排序 (rerank_async) [可选]

Stage 5: 去重 (deduplicate_async)
```

- **全异步架构**：基于 `AsyncOpenAI` + `asyncio`，零线程阻塞
- **流式输出**：SSE (Server-Sent Events) 实时返回检索进度和生成结果
- **缓存机制**：查询结果 LRU + TTL 缓存，减少重复计算

### 4. 多 LLM 故障转移

支持配置化的多级 LLM 降级链：

```
DashScope (qwen-plus) → Kimi (kimi-k2.5) → DeepSeek → 标记待处理
```

- 主 LLM 失败时自动切换备用提供商
- 指数退避重试策略
- 支持自定义降级链配置

### 5. 上下文窗口管理

- **Token 感知**：自动检测上下文是否超出 LLM 窗口限制
- **智能压缩**：保留最近 N 轮对话，压缩更早历史为摘要
- **会话持久化**：SQLite 存储多轮对话历史

## 🏗️ 技术架构

```
┌─────────────┐      ┌──────────────────────────────────────┐
│   Vue.js    │◄────►│  FastAPI                             │
│  Frontend   │      │  ├─ v1: 传统 RAG (RAGPipeline)       │
└─────────────┘      │  └─ v2: Agent 驱动 (LegalRouterAgent)│
                     └──────────────┬───────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐        ┌──────────────────┐        ┌──────────────┐
│ Query        │        │ AsyncRAGPipeline │        │ Context      │
│ Preprocessor │───────►│ ├─ Vector Search │───────►│ Manager      │
│ (分类/纠错)   │        │ ├─ BM25 Search   │        │ (Token窗口)  │
└──────────────┘        │ └─ RRF Fusion    │        └──────────────┘
                        └──────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐        ┌──────────────────┐        ┌──────────────┐
│ ChromaDB     │        │ BM25 Index       │        │ SQLite       │
│ (向量存储)    │        │ (内存索引)        │        │ (会话历史)    │
└──────────────┘        └──────────────────┘        └──────────────┘
```

### 核心模块

| 模块 | 技术栈 | 职责 |
|------|--------|------|
| **API 层** | FastAPI + Pydantic | 请求校验、依赖注入、路由管理 |
| **Agent 层** | LegalRouterAgent | 意图识别、工具路由、流程编排 |
| **检索层** | ChromaDB + BM25Okapi | 向量存储、关键词索引、混合检索 |
| **LLM 层** | AsyncOpenAI | 异步生成、流式输出、故障转移 |
| **上下文层** | ContextManager | Token 预算、历史压缩、会话管理 |

## 📊 性能特点

- **检索延迟**：< 500ms（混合检索，文档量 < 10K）
- **首 token 延迟**：< 1s（含检索 + LLM 首响应）
- **并发能力**：基于 async/await，单机支持 100+ 并发流式请求
- **缓存命中**：常用查询响应 < 50ms

## 🛠️ 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Redis 6+ (可选，用于异步任务队列)

### 安装

```bash
# 克隆项目
git clone https://github.com/driverwho/LegalRAG.git
cd LegalRAG

# 后端
cd backend
pip install -r requirements.txt

# 配置环境变量
cp app/.env.example app/.env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 前端
cd ../frontend
npm install
```

### 启动

```bash
# 启动后端
cd backend
python -m app.main

# 启动前端
cd frontend
npm run dev
```

访问：http://localhost:5173

## 💡 核心设计亮点

### 1. 意图驱动的检索策略

不同问题类型采用不同检索策略：

| 问题类型 | 检索策略 | 上下文处理 |
|---------|---------|-----------|
| **法条查询** | 向量 + BM25 混合 | 父块扩展（章/节层级） |
| **案例分析** | 向量检索为主 | 滑动窗口扩展 |
| **通用问题** | 向量检索 | 直接返回 |

### 2. 为什么用内存 BM25？

**当前方案**：ChromaDB (向量) + 内存 BM25 (关键词)

**优点**：
- 实现简单，无需额外组件
- 检索速度快（纯内存计算）
- 与向量检索天然解耦，便于分别调优

**下一步改进**：
- 文档量大时内存受限 → 迁移到 Elasticsearch 或 Milvus 稀疏向量
- 数据一致性需手动维护 → 通过事件驱动更新机制改进

### 3. 异步架构的优势

- **并行检索**：向量检索和 BM25 检索并行执行，减少总延迟
- **流式体验**：SSE 实时推送检索进度和 LLM 生成结果
- **资源高效**：单线程处理多并发，减少线程切换开销

## 📁 项目结构

```
LegalRAG/
├── backend/
│   ├── app/
│   │   ├── api/                 # API 层 (FastAPI)
│   │   │   ├── v1/              # v1 传统接口
│   │   │   │   ├── search.py    # RAG 问答
│   │   │   │   └── async_search.py  # v2 Agent 接口
│   │   │   └── async_deps.py    # 异步依赖注入
│   │   ├── core/
│   │   │   ├── agent/           # Agent 核心
│   │   │   │   ├── router.py    # LegalRouterAgent
│   │   │   │   └── tools/       # 检索工具 (法条/案例)
│   │   │   ├── retriever/       # 检索引擎
│   │   │   │   ├── async_rag.py # 5阶段异步流水线
│   │   │   │   ├── bm25.py      # BM25 实现
│   │   │   │   └── fusion.py    # RRF 融合
│   │   │   ├── preprocessor/    # 查询预处理
│   │   │   │   ├── query_preprocessor.py
│   │   │   │   └── classifier.py
│   │   │   └── context/         # 上下文管理
│   │   └── main.py              # 应用入口
│   └── requirements.txt
├── frontend/                    # Vue.js 前端
└── README.md
```

## 📄 License

MIT License

---

**技术栈**: FastAPI + Vue3 + ChromaDB + LangChain + BM25Okapi + PaddleOCR + SQLite + Celery/Redis
**架构版本**: v2.0 (Agent-driven)
