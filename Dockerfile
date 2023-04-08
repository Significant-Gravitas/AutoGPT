FROM python:3.11

WORKDIR /app
COPY auto-gpt/ /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
