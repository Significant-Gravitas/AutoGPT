FROM python:3.11

RUN pip install poetry

WORKDIR /app

COPY poetry.lock /app
COPY pyproject.toml /app
RUN poetry install

COPY scripts/ /app

CMD ["poetry", "run", "python", "main.py"]
