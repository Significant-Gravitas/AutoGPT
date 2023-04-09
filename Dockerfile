FROM python:3.11

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY autogpt/ /app/autogpt

ENTRYPOINT ["python", "-m", "autogpt"]
