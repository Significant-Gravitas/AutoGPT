FROM python:3.11-slim-buster AS server_base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# postgresql-client is needed to check if the postgres service is ready for running migrations
RUN apt-get update \
    && apt-get install -y build-essential curl ffmpeg wget libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev postgresql-client \
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

FROM server_base AS server_dependencies

RUN mkdir -p /app/autogpt
RUN mkdir -p /app/forge
RUN mkdir -p /app/rnd/autogpt_libs
RUN mkdir -p /app/rnd/autogpt_server

COPY autogpt /app/autogpt
COPY forge /app/forge
COPY rnd/autogpt_libs /app/rnd/autogpt_libs

COPY rnd/autogpt_server/poetry.lock rnd/autogpt_server/pyproject.toml /app/rnd/autogpt_server/

WORKDIR /app/rnd/autogpt_server

RUN poetry install --no-interaction --no-ansi

FROM server_dependencies AS server_prisma

COPY rnd/autogpt_server/schema.prisma ./
RUN poetry run prisma generate

FROM server_prisma AS server

COPY rnd/autogpt_server /app/rnd/autogpt_server

ENV DATABASE_URL=""
ENV PORT=8000

CMD ["poetry", "run", "rest"]

