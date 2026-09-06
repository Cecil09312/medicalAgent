# ============================================================
# AI智能医疗诊断系统 - Dockerfile
# 使用国内镜像加速构建（pip 清华源 / apt 清华源 / HuggingFace 国内镜像）
# ============================================================

# 基础镜像（通过 DaoCloud 国内镜像拉取官方 python:3.12-slim）
FROM docker.m.daocloud.io/library/python:3.12-slim

# ------------------------------------------------------------
# 环境变量
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    # HuggingFace 国内镜像（模型下载走 hf-mirror.com）
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_OFFLINE=0 \
    # pip 不缓存，减小镜像体积
    PIP_NO_CACHE_DIR=1

# ------------------------------------------------------------
# 配置 apt 国内镜像（清华大学开源软件镜像站）
# ------------------------------------------------------------
RUN sed -i \
    -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
    -e 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
    /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i \
        -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
        -e 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
        /etc/apt/sources.list 2>/dev/null \
    || true

# 安装系统编译依赖（部分 Python 包需要 gcc/g++ 编译）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 工作目录
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# 配置 pip 国内镜像（清华大学 PyPI 镜像）并升级 pip
# ------------------------------------------------------------
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ------------------------------------------------------------
# 安装 Python 依赖
# 先单独拷贝 requirements.txt，利用 Docker 缓存层，避免代码变更重复安装依赖
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ------------------------------------------------------------
# 拷贝项目代码
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# 创建运行时数据目录
# ------------------------------------------------------------
RUN mkdir -p /app/logs \
    /app/review/data \
    /app/flywheel/data \
    /app/knowledge/data \
    /app/evaluation/data/reports

# 授予启动脚本执行权限
RUN chmod +x /app/docker-entrypoint.sh

# ------------------------------------------------------------
# 暴露 Web 服务端口
# ------------------------------------------------------------
EXPOSE 7860

# ------------------------------------------------------------
# 启动入口：自动初始化知识库 + 启动 Web 服务
# ------------------------------------------------------------
ENTRYPOINT ["/app/docker-entrypoint.sh"]
