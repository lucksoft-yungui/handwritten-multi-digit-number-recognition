# 第一阶段：构建Python环境并安装依赖
FROM python:3.8 AS builder

WORKDIR /app

RUN apt-get update && apt-get -y install cmake
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# 复制 requirements.txt 文件并安装依赖
COPY requirements/prod.txt ./requirements.txt
RUN pip install -r requirements.txt

# 第二阶段：复制代码和目录
FROM builder

WORKDIR /app

# 从第一个阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.8/site-packages/ /usr/local/lib/python3.8/site-packages/

# 复制应用程序代码和目录
COPY handwritten_multi_digit_number_recognition ./handwritten_multi_digit_number_recognition/
COPY images ./images/
COPY artifacts ./artifacts/
COPY rec.py .

# 设置容器启动命令
# CMD ["python", "your_script.py"]