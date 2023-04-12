FROM python:3.11

WORKDIR /app
COPY scripts/ /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
