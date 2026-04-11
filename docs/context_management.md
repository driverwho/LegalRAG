# 上下文管理实现指南

## 概述

上下文管理（Context Management）用于在 RAG 系统中支持多轮对话，让 LLM 能够理解对话历史并回答指代性问题（如"它是什么意思？"、"再详细说说"）。

## 当前状态

当前系统是无状态 RAG：
- 每次查询独立，仅使用检索的文档作为上下文
- 不加载对话历史
- 不支持多轮对话中的指代理解

## 新增组件

### 1. ContextManager (`backend/app/core/context/`)

负责管理对话历史：
- 从 SQLite 加载历史消息
- 管理 Token 预算和上下文窗口
- 超长对话时进行摘要压缩
- 格式化消息供 LLM 使用

**配置选项：**
```python
ContextConfig(
    max_context_messages=10,      # 保留最近多少条消息
    max_tokens_per_message=500,   # 单条消息最大长度
    enable_summarization=True,    # 是否启用摘要
    summarization_threshold=20,   # 超过多少条时摘要
)
```

### 2. ContextualChatManager (`backend/app/core/llm/contextual_chat.py`)

扩展的 ChatManager，支持对话上下文：
- 在生成回答时融入历史对话
- 保持与原有接口兼容
- 无 session_id 时退化为基础版本

### 3. ContextualRAGPipeline (`backend/app/core/retriever/contextual_rag.py`)

扩展的 RAGPipeline，传递 session_id 到生成阶段。

## 集成步骤

### 步骤 1: 启用上下文管理（已创建文件）

已创建以下文件：
- `backend/app/core/context/context_manager.py`
- `backend/app/core/context/__init__.py`
- `backend/app/core/llm/contextual_chat.py`
- `backend/app/core/retriever/contextual_rag.py`

### 步骤 2: 更新依赖注入

在 `backend/app/api/deps.py` 中添加：

```python
from backend.app.core.context import ContextManager
from backend.app.core.llm.contextual_chat import ContextualChatManager
from backend.app.core.retriever.contextual_rag import ContextualRAGPipeline

@lru_cache()
def get_context_manager() -> ContextManager:
    session_service = get_session_service()
    return ContextManager(session_service)

@lru_cache()
def get_contextual_chat_manager() -> ContextualChatManager:
    settings = get_settings()
    context_manager = get_context_manager()
    return ContextualChatManager(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        context_manager=context_manager,
    )

@lru_cache()
def get_contextual_rag_pipeline() -> ContextualRAGPipeline:
    settings = get_settings()
    return ContextualRAGPipeline(
        vector_store=get_vector_store(),
        chat_manager=get_contextual_chat_manager(),
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        max_results=settings.MAX_RESULTS,
    )
```

### 步骤 3: 修改 Query 端点

修改 `backend/app/api/v1/search.py`：

```python
from backend.app.core.retriever.contextual_rag import ContextualRAGPipeline
from backend.app.api.deps import get_contextual_rag_pipeline

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: SessionQueryRequest,
    pipeline: ContextualRAGPipeline = Depends(get_contextual_rag_pipeline),
    session_service: SessionService = Depends(get_session_service),
):
    result = pipeline.answer(
        question=body.question,
        k=body.k,
        collection_name=body.collection_name,
        session_id=body.session_id,  # 传递 session_id 启用上下文
    )
    
    # 保存消息（原有逻辑）...
    
    return QueryResponse(...)
```

### 步骤 4: 前端无需修改

前端已传递 `session_id`，无需改动。

## 工作流程

```
用户提问 -> 检索文档 -> 加载历史消息 -> 组装完整 Prompt -> LLM 生成
                |                                    |
                v                                    v
          向量数据库搜索                          滑动窗口选择
          (相似度 top-k)                          (最近 N 条)
```

## Prompt 结构

```
System: RAG 指令 + 检索到的文档内容
User 1: 第一问
Assistant 1: 第一答
User 2: 第二问
Assistant 2: 第二答
...
User N: 当前问题（与检索上下文一起）
```

## 注意事项

1. **Token 消耗增加**：加入历史对话会增加 Token 使用量
2. **需要足够的上下文窗口**：确保模型支持足够的 Token 长度
3. **隐私考虑**：历史对话会作为 Prompt 发送给 LLM
4. **性能**：加载历史消息会增加少许延迟（SQLite 查询）

## 可选优化

1. **智能摘要**：对超长对话使用 LLM 生成摘要而非简单截断
2. **关键信息提取**：提取实体和关键信息替代完整对话
3. **分层上下文**：近期详细 + 远期摘要的分层策略
4. **上下文相关性过滤**：只加载与当前问题相关的历史消息

## 回退方案

如果不需要上下文管理，保持现有代码即可：
- 不传 `session_id` 时自动使用基础版本
- 可随时切换回 `RAGPipeline` 和 `ChatManager`
