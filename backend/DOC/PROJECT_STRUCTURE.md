# 项目结构规范

## 项目整体结构

```
vector_databases/
├── backend/                           # 后端代码目录
│   ├── __init__.py
│   ├── app/                          # Flask应用目录
│   │   ├── __init__.py
│   │   ├── server.py                 # Flask应用入口
│   │   ├── routes/                   # API路由目录
│   │   │   ├── __init__.py
│   │   │   ├── vector_routes.py      # 向量数据库相关路由
│   │   │   └── general_routes.py     # 通用路由
│   │   └── config.py                 # 应用配置
│   ├── core/                         # 核心模块目录
│   │   ├── __init__.py
│   │   ├── vector_db_manager.py      # 向量数据库管理器
│   │   ├── vector_retriever.py       # 向量检索器
│   │   ├── document_loader.py        # 文档加载器
│   │   └── embeddings/               # 嵌入模型相关
│   │       ├── __init__.py
│   │       └── dashscope_embeddings.py
│   ├── utils/                        # 工具函数目录
│   │   ├── __init__.py
│   │   ├── file_utils.py             # 文件处理工具
│   │   └── validation_utils.py       # 验证工具
│   ├── tests/                        # 后端测试目录
│   │   ├── __init__.py
│   │   ├── test_vector_db.py         # 向量数据库测试
│   │   └── test_api_endpoints.py     # API端点测试
│   └── requirements.txt              # Python依赖
├── frontend/                         # 前端代码目录
│   ├── public/                       # 静态资源
│   ├── src/                          # 源代码
│   │   ├── assets/                   # 静态资源
│   │   ├── components/               # Vue组件
│   │   │   ├── common/               # 通用组件
│   │   │   ├── rag/                  # RAG相关组件
│   │   │   └── layout/               # 布局组件
│   │   ├── views/                    # 页面视图
│   │   ├── router/                   # 路由配置
│   │   ├── store/                    # 状态管理
│   │   ├── services/                 # API服务
│   │   ├── utils/                    # 工具函数
│   │   ├── styles/                   # 样式文件
│   │   ├── App.vue                   # 根组件
│   │   └── main.js                   # 入口文件
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js                # 构建配置
│   └── README.md
├── docs/                             # 文档目录
│   ├── installation.md               # 安装指南
│   ├── api_documentation.md          # API文档
│   ├── usage_guide.md                # 使用指南
│   └── architecture.md               # 架构说明
├── data/                             # 数据目录
│   ├── uploads/                      # 上传文件存储
│   └── chroma_db/                    # ChromaDB数据存储
├── scripts/                          # 脚本目录
│   ├── setup_dev_env.sh              # 开发环境设置脚本
│   ├── setup_dev_env.bat             # Windows开发环境设置脚本
│   └── deploy.sh                     # 部署脚本
├── tests/                            # 顶层测试目录
│   └── integration_tests/            # 集成测试
├── .env.example                      # 环境变量示例
├── .gitignore                        # Git忽略文件
├── docker/                           # Docker相关文件
│   ├── Dockerfile.backend            # 后端Dockerfile
│   ├── Dockerfile.frontend           # 前端Dockerfile
│   └── docker-compose.yml            # Docker Compose配置
├── README.md                         # 项目主说明文件
└── LICENSE                           # 许可证文件
```

## 主要改进点

1. **清晰分离前后端代码**：分别置于`backend/`和`frontend/`目录下
2. **模块化组织**：后端代码按功能分为app、core、utils等目录
3. **标准化命名**：使用一致的目录和文件命名约定
4. **测试组织**：将测试代码合理分布在不同层级
5. **文档集中管理**：所有文档集中在`docs/`目录下
6. **数据隔离**：将用户上传的数据和数据库存储分离到`data/`目录

## 开发规范

### 后端开发规范

- 所有Python代码遵循PEP8规范
- 使用类型注解增强代码可读性
- 每个模块需要包含适当的文档字符串
- 错误处理和日志记录要一致

### 前端开发规范

- Vue组件使用单文件组件(SFC)格式
- 组件命名使用PascalCase
- 使用ESLint和Prettier进行代码格式化
- 遵循Vue 3 Composition API风格

### Git工作流

- 主分支(master/main)：生产就绪代码
- 开发分支(develop)：集成准备
- 功能分支(feature/*)：新功能开发
- 修复分支(hotfix/*)：紧急修复