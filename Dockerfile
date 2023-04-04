FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade --no-cache-dir pip \
&&  pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "scripts/main.py"]
