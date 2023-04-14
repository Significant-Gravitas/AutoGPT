FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=yes
RUN apt update && apt install -y curl
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ .
ENTRYPOINT ["python", "main.py"]
