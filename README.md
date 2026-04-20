# LegalRAG - 法律智能问答系统

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.5+-4FC08D.svg?logo=vue.js&logoColor=white)](https://vuejs.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-FF6F00.svg)](https://www.trychroma.com)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.0+-2B6CB0.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B35.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个面向法律领域的** Agent 驱动 RAG (Retrieval-Augmented Generation) 系统**，支持意图识别、混合检索、流式输出和多级 LLM 故障转移。

> **核心定位**：通过意图路由和混合检索，实现法律文档的智能问答与精准引用。

## 🚀 核心特性

### 1. Agent 驱动的智能路由

系统提供两代 Agent 实现，共享相同的 `run_stream()` / `run()` 接口：

#### v2 — 静态路由 (`LegalRouterAgent`)

基于 `_TOOL_MAP` 的确定性单轮路由，快速高效：

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

#### v3 — ReAct 推理循环 (`LegalReActAgent`)

基于 **LangGraph** 的 ReAct (Reasoning + Acting) Agent，由 LLM 自主决定工具调用，支持多轮推理与自我纠错：

```
┌──────────────┐
│  preprocess   │  QueryPreprocessor (分类/纠错，图外执行)
└──────┬────────┘
       │
┌──────▼────────┐
│    reason      │◄────────┐  LLM 决策：调用工具 or 输出最终回答
└──────┬────────┘         │
       │ tool_calls        │ 循环
 ┌─────▼──────┐    ┌──────┴────────┐
 │    act      │───►│  ToolNode     │  执行工具，追加 ToolMessage
 │ (ToolNode)  │    │  返回结果     │
 └─────────────┘    └───────────────┘
       │ 无更多 tool_calls
┌──────▼────────┐
│     END        │  最终 AI 回答
└───────────────┘
```

**与 v2 的核心差异**：

| 特性 | v2 (LegalRouterAgent) | v3 (LegalReActAgent) |
|------|----------------------|---------------------|
| **工具选择** | 静态 `_TOOL_MAP` 映射 | LLM 自主决策 |
| **推理轮次** | 单轮 | 多轮 (最多 `max_iterations` 次) |
| **自我纠错** | 不支持 | 检索不足时自动换关键词重试 |
| **复杂度适应** | 固定策略 | 根据 complexity hint 动态调整检索深度 |
| **技术栈** | 自定义路由逻辑 | LangGraph `StateGraph` + `ToolNode` |

**v3 新增 SSE 事件类型**：

| 事件 | 说明 |
|------|------|
| `tool_dispatch` | LLM 选择了哪些工具 (每轮迭代) |
| `observation` | 工具执行结果摘要 (截断至 600 字) |

**v3 关键设计**：

- **AgentState**：基于 `TypedDict` 的 LangGraph 状态，通过 `add_messages` reducer 自动追加消息
- **ReAct Prompt**：工具无关的系统提示词 (工具 schema 由 `bind_tools` 自动注入)，按复杂度动态注入效率提示
- **迭代控制**：`max_iterations=5` 硬上限，防止无限循环和成本失控
- **API 端点**：`POST /query/stream/v3` (流式) / `POST /query/v3` (非流式)，与 v2 接口 Drop-in 兼容

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

### 6. 检索性能评估 (V2)

基于两个开源法律基准数据集，对检索流水线**各阶段独立评估**，量化每个模块的贡献并定位性能瓶颈。

#### 6.1 评估数据集

| # | 数据集 | 规模 | 格式 | 来源 |
|---|--------|------|------|------|
| 1 | 带法律依据的情景问答 | 92K | QA（问题 + 法条引用 + 参考答案） | [LAW-GPT](https://github.com/LiuHC0428/LAW-GPT?tab=readme-ov-file) |
| 2 | 法律单轮问答 | 52K | MCQ（选择题 + 解析 + 正确答案） | [lawyer-llama](https://github.com/AndrewZhe/lawyer-llama) |

- **数据集 1 (QA)**：每条样本包含用户问题、标准法条引用列表（ground truth）和参考答案，直接用于评估检索召回与生成质量
- **数据集 2 (MCQ)**：从题目中提取纯问题部分送入检索流水线，从解析文本中提取引用的法条名称作为 ground truth 评估检索能力

#### 6.2 评估模块设计

评估系统位于 `backend/app/core/evaluation/`，采用**逐阶段截断评估**架构：复用生产环境的 `AsyncRAGPipeline` 各阶段方法，在每个阶段输出后独立计算指标。

```
backend/app/core/evaluation/
├── datasets.py              # 数据集加载器（QA/MCQ 双格式自动检测）
├── metrics.py               # 指标计算引擎（Recall/MRR/NDCG/ROUGE-L）
├── pipeline_evaluator.py    # 核心：逐阶段流水线评估器
├── report.py                # 报告输出（JSON + Markdown 表格）
└── run_evaluation.py        # CLI 入口
```

**逐阶段评估流程**：

```
Stage 1  查询预处理        → 记录分类结果、耗时、分类置信度
Stage 2a 向量检索 (独立)   → Recall@K / Precision@K / MRR / NDCG@K
Stage 2b BM25 检索 (独立)  → 同上，用于对比语义检索 vs 关键词检索
Stage 2c RRF 融合          → 同上，量化融合增益 (Δ Recall)
Stage 3  上下文组装        → 父块扩展/滑动窗口后的指标变化
Stage 4  重排序 (可选)     → 重排序对排名质量的提升
Stage 5  去重              → 最终检索指标
Stage 6  LLM 生成 (端到端) → ROUGE-L / Jaccard 相似度 / 法条引用准确率
```

**评估指标体系**：

| 类别 | 指标 | 说明 |
|------|------|------|
| 检索质量 | **Recall@K** | top-K 结果中命中的 ground truth 比例 |
| 检索质量 | **Precision@K** | top-K 结果中相关文档的比例 |
| 检索质量 | **F1@K** | Precision 与 Recall 的调和平均 |
| 排名质量 | **MRR** | 第一个相关文档排名的倒数均值 |
| 排名质量 | **NDCG@K** | 归一化折损累积增益 |
| 覆盖率 | **Hit Rate@K** | 至少命中一个 ground truth 的样本比例 |
| 生成质量 | **ROUGE-L** | 基于最长公共子序列的文本重叠度 |
| 生成质量 | **Jaccard** | jieba 分词后的词级 Jaccard 相似度 |
| 生成质量 | **Citation Acc** | 生成答案中正确引用法条的比例 |

**相关性判断策略**：采用双向子串匹配 + jieba 分词 Jaccard 重叠（阈值 30%），适配法律文本中法条编号精确、内容片段冗长的特点。

#### 6.3 运行方式

```bash
# 基础评估（仅检索指标）
python -m backend.app.core.evaluation.run_evaluation \
    --dataset path/to/qa_dataset.json \
    --collection-name law_collection

# 完整评估（检索 + 生成）
python -m backend.app.core.evaluation.run_evaluation \
    --dataset qa.json --dataset mcq.json \
    --output-dir ./evaluation_results \
    --enable-generation \
    --k-values 1,3,5,10

# 快速测试（限制样本数）
python -m backend.app.core.evaluation.run_evaluation \
    --dataset qa.json \
    --max-samples 50 \
    --no-generation
```


**关键结论**：

- **法条查询准确率**：全链路的查询准确率达到**73%**
- **RRF 融合增益**：相比单一检索器，Recall@5 预计提升 10~15 个百分点。向量检索擅长语义匹配（如"被告人不服判决的权利"→ "上诉权"），BM25 擅长精确匹配（如法条编号"第二百九十四条"），两者互补
- **上下文组装影响**：父块扩展会降低 Precision（引入更多上下文），但保持 Recall 不变，有利于 LLM 生成更完整的答案
- **去重提升 Precision**：去除冗余后 Precision 提升**5.2%**
- **MCQ 数据集的挑战**：选择题的 ground truth 需从解析文本中提取法条引用，提取质量直接影响评估准确性；Hit Rate@5 略低于 QA 数据集

- 剩余数据指标待进一步完善。


## 🏗️ 技术架构

```
┌─────────────┐      ┌──────────────────────────────────────────────────┐
│   Vue.js    │◄────►│  FastAPI                                         │
│  Frontend   │      │  ├─ v1: 传统 RAG (RAGPipeline)                   │
│             │      │  ├─ v2: Agent 静态路由 (LegalRouterAgent)         │
└─────────────┘      │  └─ v3: ReAct 推理循环 (LegalReActAgent)│
                     └──────────────┬───────────────────────────────────┘
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
| **Agent 层 (v2)** | LegalRouterAgent | 静态意图识别、工具路由、流程编排 |
| **Agent 层 (v3)** | LegalReActAgent + LangGraph | ReAct 推理循环、LLM 自主工具选择、多轮推理 |
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
│   │   │   │   └── async_search.py  # v2/v3 Agent 接口
│   │   │   └── async_deps.py    # 异步依赖注入
│   │   ├── core/
│   │   │   ├── agent/           # Agent 核心
│   │   │   │   ├── router.py    # LegalRouterAgent (v2 静态路由)
│   │   │   │   ├── react_agent.py # LegalReActAgent (v3 ReAct) │   │   │   │   ├── state.py     # AgentState (LangGraph 状态定义)
│   │   │   │   ├── tools/       # 检索工具 (法条/案例)
│   │   │   │   │   └── base.py  # AgentTool 基类 + LangChain 适配
│   │   │   │   └── prompts/     # Prompt 模板
│   │   │   │       ├── registry.py      # 分类型 prompt 注册
│   │   │   │       └── react_prompts.py # v3 ReAct prompt
│   │   │   ├── retriever/       # 检索引擎
│   │   │   │   ├── async_rag.py # 5阶段异步流水线
│   │   │   │   ├── bm25.py      # BM25 实现
│   │   │   │   └── fusion.py    # RRF 融合
│   │   │   ├── evaluation/      # V2 检索性能评估
│   │   │   │   ├── datasets.py  # 数据集加载器
│   │   │   │   ├── metrics.py   # 评估指标计算
│   │   │   │   ├── pipeline_evaluator.py  # 逐阶段评估器
│   │   │   │   ├── report.py    # 报告生成
│   │   │   │   └── run_evaluation.py  # CLI 入口
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

## 🔭 进一步改进方向

### 检索质量

| 方向 | 现状 | 改进方案 | 预期收益 |
|------|------|----------|----------|
| **Embedding 模型** | DashScope text-embedding-v3 | 引入 BGE-M3 / GTE 等开源模型做 A/B 对比 | 降低 API 依赖，可离线部署 |
| **稀疏-稠密联合索引** | 内存 BM25 + ChromaDB 分离 | 迁移至 Milvus (稀疏+稠密混合索引) 或 Elasticsearch 8.x (kNN + BM25) | 单引擎混合检索，简化架构 |
| **查询改写** | 单次查询 | 增加 HyDE（假设性文档嵌入）+ 多查询扩展 | 提升语义召回率 |
| **分块策略** | 固定分块 + 父子关联 | 引入语义分块（按法条条款/案例段落边界切分） | 减少截断噪声，提升上下文完整性 |

### 工程化与可观测性

- [x] **Docker Compose 一键部署**：后端 + 前端 + ChromaDB + Redis 容器化编排，消除环境差异
- [ ] **CI/CD 流水线**：GitHub Actions 自动化测试 + 镜像构建 + 部署
- [x] **检索质量评估**：V2 逐阶段评估框架已实现，支持 QA/MCQ 双数据集、Recall@K / MRR / NDCG / ROUGE-L 全指标覆盖
- [ ] **Prometheus + Grafana 监控**：API 延迟 P50/P99、向量库容量、BM25 索引大小、LLM 故障转移次数

### 功能扩展

- [x] **多轮推理 Agent**：v3 `LegalReActAgent` 已实现基于 LangGraph 的 ReAct 推理循环，支持 LLM 自主多轮工具调用和自我纠错
- [ ] **用户反馈闭环**：引入 thumbs up/down 机制，收集用户对检索结果和回答质量的反馈，用于持续优化检索阈值和 prompt
- [ ] **知识图谱增强**：构建法律条文间的引用关系图（"本法第 X 条"→ 自动关联），支持图谱辅助检索

### 生产就绪

- [ ] **API 鉴权与限流**：JWT 认证 + 基于令牌桶的速率限制，防止滥用
- [ ] **向量增量更新**：新法律文件入库时仅嵌入增量部分，避免全量重建索引
- [ ] **多模型路由策略**：根据问题复杂度动态选择模型（简单问题走轻量模型降本，复杂问题走大模型保质量）

---

**技术栈**: FastAPI + Vue3 + ChromaDB + LangChain + LangGraph + BM25Okapi + PaddleOCR + SQLite + Celery/Redis
**架构版本**: v3.0 (ReAct Agent, 默认) / v2.0 (Agent-driven, 可配置回退)
