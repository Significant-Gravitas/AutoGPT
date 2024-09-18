FROM python:3.11-slim-buster AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y build-essential curl ffmpeg wget libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev postgresql-client git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$POETRY_HOME/bin:$PATH"

# Upgrade pip and setuptools to fix security vulnerabilities
RUN pip3 install --upgrade pip setuptools

RUN pip3 install poetry

# Copy and install dependencies
COPY rnd/autogpt_libs /app/rnd/autogpt_libs
COPY rnd/autogpt_server/poetry.lock rnd/autogpt_server/pyproject.toml /app/rnd/autogpt_server/
WORKDIR /app/rnd/autogpt_server
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Generate Prisma client
COPY rnd/autogpt_server/schema.prisma ./
RUN poetry config virtualenvs.create false \
    && poetry run prisma generate

FROM python:3.11-slim-buster AS server_dependencies

WORKDIR /app

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="$POETRY_HOME/bin:$PATH"


# Upgrade pip and setuptools to fix security vulnerabilities
RUN pip3 install --upgrade pip setuptools

# Copy only necessary files from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy Prisma binaries
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries


ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /app/rnd/autogpt_libs
RUN mkdir -p /app/rnd/autogpt_server

COPY rnd/autogpt_libs /app/rnd/autogpt_libs

COPY rnd/autogpt_server/poetry.lock rnd/autogpt_server/pyproject.toml /app/rnd/autogpt_server/

WORKDIR /app/rnd/autogpt_server

FROM server_dependencies AS server

COPY rnd/autogpt_server /app/rnd/autogpt_server

ENV DATABASE_URL=""
ENV PORT=8000

CMD ["poetry", "run", "rest"]

