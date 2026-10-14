# 使用官方 Python 基础镜像（和本地版本保持一致，3.14）
FROM python:3.14-slim

# 设置容器内的工作目录
WORKDIR /app

# 先单独复制依赖清单并安装
# （放前面是为了利用 Docker 层缓存：只改代码不装依赖时，这一步会跳过，构建更快）
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 复制项目其余代码到工作目录
COPY . .

# 声明容器对外暴露的端口
EXPOSE 8000

# 容器启动时执行的命令
# 注意：必须用 --host 0.0.0.0，否则容器外（宿主机）访问不到
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
