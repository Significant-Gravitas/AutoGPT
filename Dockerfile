FROM python:3.11


WORKDIR /tmp
COPY requirements.txt /tmp
RUN pip install -r requirements.txt

WORKDIR /app
COPY scripts/ /app

CMD ["python", "main.py"]
