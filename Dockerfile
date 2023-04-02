FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade --no-cache-dir pip \
&&  pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "main.py"]
