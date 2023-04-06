FROM python:3.11

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

CMD ["python", "scripts/main.py"]

