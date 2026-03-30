# Milvus 向量数据库设置指南

## 方法一：使用 Docker Compose（推荐）

### 1. 安装 Docker 和 Docker Compose
- 确保已安装 Docker Desktop 或 Docker Engine
- Windows 用户请启用 WSL 2 或使用 Linux 虚拟机

### 2. 启动 Milvus 服务
在项目根目录运行以下命令：

```bash
docker-compose up -d
```

### 3. 检查服务状态
```bash
docker-compose ps
```

所有服务状态应该是 `healthy`。

### 4. 访问服务
- MinIO 管理界面: http://localhost:9001
  - 账号: `minioadmin`
  - 密码: `minioadmin`
- Milvus 服务端口: `localhost:19530`

### 5. 停止服务
```bash
docker-compose down
```

## 方法二：使用 Docker Run 命令

如果不想使用 docker-compose，也可以用单个命令启动 Milvus：

```bash
# 启动依赖服务
docker run -d --name milvus-etcd -p 2379:2379 -e ETCD_AUTO_COMPACTION_MODE=revision -e ETCD_AUTO_COMPACTION_RETENTION=1000 -e ETCD_QUOTA_BACKEND_BYTES=4294967296 -e ETCD_SNAPSHOT_COUNT=50000 quay.io/coreos/etcd:v3.5.5 /usr/local/bin/etcd -advertise-client-urls=http://0.0.0.0:2379 -listen-client-urls=http://0.0.0.0:2379 -initial-cluster=default=http://0.0.0.0:2380 -listen-peer-urls=http://0.0.0.0:2380 -initial-advertise-peer-urls=http://0.0.0.0:2380 -initial-cluster-token=etcd-cluster-1 -initial-cluster-state=new -data-dir=/etcd

docker run -d --name milvus-minio -p 9000:9000 -p 9001:9001 -e MINIO_ACCESS_KEY=minioadmin -e MINIO_SECRET_KEY=minioadmin minio/minio:RELEASE.2023-09-04T19-57-37Z server /minio_data --console-address ":9001"

# 等待以上两个容器启动完成后再运行Milvus
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 --link milvus-etcd:etcd --link milvus-minio:minio -e ETCD_ENDPOINTS=etcd:2379 -e MINIO_ADDRESS=minio:9000 milvusdb/milvus:v2.3.1 milvus run standalone
```

## 方法三：云托管的 Milvus

如果本地部署困难，可以考虑使用 Zilliz Cloud 提供的托管 Milvus 服务：

1. 访问 https://cloud.zilliz.com/
2. 注册账户并创建免费集群
3. 获取连接信息并更新项目的 `.env` 文件

## 故障排除

### 服务启动失败
- 检查端口是否已被占用
- 确保 Docker 有足够的资源分配
- 查看容器日志：`docker logs <container_name>`

### 连接超时
- 确认 Milvus 服务健康状态：`docker-compose ps`
- 检查防火墙设置
- 确认 `.env` 文件中的连接参数

### 存储空间不足
- 清理 Docker 系统缓存：`docker system prune`
- 检查磁盘空间：`df -h`

## 环境变量配置

确保项目根目录的 `.env` 文件包含正确的连接信息：

```
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
```