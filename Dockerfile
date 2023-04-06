FROM python:3.11

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

CMD ["python", "scripts/main.py"]
