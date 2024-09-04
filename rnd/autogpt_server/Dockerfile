FROM python:3.11-slim-buster as server_base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


WORKDIR /app

RUN apt-get update \
    && apt-get install -y build-essential curl ffmpeg wget libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://github.com/git/git/archive/v2.28.0.tar.gz -O git.tar.gz \
    && tar -zxf git.tar.gz \
    && cd git-* \
    && make prefix=/usr all \
    && make prefix=/usr install


ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$POETRY_HOME/bin:$PATH"
RUN pip3 install poetry

COPY autogpt /app/autogpt
COPY forge /app/forge
COPY rnd/autogpt_libs /app/rnd/autogpt_libs

WORKDIR /app/rnd/autogpt_server

COPY rnd/autogpt_server/pyproject.toml rnd/autogpt_server/poetry.lock ./
RUN poetry install --no-interaction --no-ansi

COPY rnd/autogpt_server/schema.prisma ./
RUN poetry run prisma generate

COPY rnd/autogpt_server /app/rnd/autogpt_server
FROM server_base as server

ENV PORT=8000
ENV DATABASE_URL=""

CMD ["poetry", "run", "app"]
