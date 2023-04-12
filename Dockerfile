FROM python:3.11

RUN pip install poetry

WORKDIR /app

COPY poetry.lock pyproject.toml /app/
RUN poetry install

COPY scripts/ /app
COPY requirements.txt /app

CMD ["poetry", "run", "python", "main.py"]
