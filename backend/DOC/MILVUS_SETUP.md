# ChromaDB 向量数据库设置指南

## 简介

ChromaDB 是一个轻量级、易于使用的开源向量数据库，非常适合本地开发和小型项目。与 Milvus 相比，ChromaDB 不需要复杂的部署，可以直接作为 Python 库使用。

## 安装与配置

### 1. 安装依赖
项目依赖已在 `requirements.txt` 中定义，运行以下命令安装：

```bash
pip install -r requirements.txt
```

### 2. 配置环境
环境变量已在 `.env` 文件中定义，包含以下参数：

```
# ChromaDB Configuration
CHROMADB_PERSIST_DIR=./chroma_db
COLLECTION_NAME=agent_rag

# DashScope / LLM Configuration
DASHSCOPE_API_KEY=sk-f582f548fab146928eb409e6411ae627
EMBEDDING_MODEL=text-embedding-v3
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# Flask Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
```

### 3. 启动服务
直接运行后端服务即可，ChromaDB 会在需要时自动初始化：

```bash
python server.py
```

## 使用说明

### 数据持久化
ChromaDB 默认会将数据保存到 `./chroma_db` 目录中，重启服务后数据不会丢失。

### 集合管理
- 集合会在首次使用时自动创建
- 集合名称在环境变量 `COLLECTION_NAME` 中定义
- 删除集合可以通过后台管理接口或直接删除持久化目录中的相应文件夹

## 故障排除

### 服务启动失败
- 检查依赖是否正确安装：`pip install -r requirements.txt`
- 确认 `.env` 文件中的路径和参数配置正确

### 连接超时
- 确认 `.env` 文件中的连接参数
- 检查网络连接和防火墙设置

### 存储空间不足
- 清理 ChromaDB 持久化目录：删除 `./chroma_db` 目录（注意：这会清除所有数据）
- 检查磁盘空间：`df -h`