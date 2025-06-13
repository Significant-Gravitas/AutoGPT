FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install .
EXPOSE 6969
CMD ["python", "-m", "solenne.api"]
