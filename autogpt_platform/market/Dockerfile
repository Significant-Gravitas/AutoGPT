FROM python:3.11.10-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN echo 'Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;\n' > /etc/apt/apt.conf.d/99fixbadproxy

RUN apt-get update --allow-releaseinfo-change --fix-missing

# Install build dependencies
RUN apt-get install -y build-essential 
RUN apt-get install -y libpq5
RUN apt-get install -y libz-dev
RUN apt-get install -y libssl-dev

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

# Upgrade pip and setuptools to fix security vulnerabilities
RUN pip3 install --upgrade pip setuptools

RUN pip3 install poetry

# Copy and install dependencies
COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs
COPY autogpt_platform/market/poetry.lock autogpt_platform/market/pyproject.toml /app/autogpt_platform/market/
WORKDIR /app/autogpt_platform/market
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Generate Prisma client
COPY autogpt_platform/market /app/autogpt_platform/market
RUN poetry config virtualenvs.create false \
    && poetry run prisma generate

FROM python:3.11.10-slim-bookworm AS server_dependencies

WORKDIR /app

# Upgrade pip and setuptools to fix security vulnerabilities
RUN pip3 install --upgrade pip setuptools

# Copy only necessary files from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy Prisma binaries
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries

ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /app/autogpt_platform/autogpt_libs
RUN mkdir -p /app/autogpt_platform/market

COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs

COPY autogpt_platform/market /app/autogpt_platform/market

WORKDIR /app/autogpt_platform/market

FROM server_dependencies AS server

ENV DATABASE_URL=""
ENV PORT=8015

CMD ["poetry", "run", "app"]
