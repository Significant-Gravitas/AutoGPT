FROM debian:13-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN echo 'Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;\n' > /etc/apt/apt.conf.d/99fixbadproxy

# Update package list and install Python and build dependencies
RUN apt-get update --allow-releaseinfo-change --fix-missing \
    && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    build-essential \
    libpq5 \
    libz-dev \
    libssl-dev \
    postgresql-client

ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH=/opt/poetry/bin:$PATH

# Upgrade pip and setuptools to fix security vulnerabilities
RUN python3 -m pip install --upgrade pip setuptools --break-system-packages

RUN python3 -m pip install poetry

# Copy and install dependencies
COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs
COPY autogpt_platform/backend/poetry.lock autogpt_platform/backend/pyproject.toml /app/autogpt_platform/backend/
WORKDIR /app/autogpt_platform/backend
RUN poetry install --no-ansi --no-root

# Generate Prisma client
COPY autogpt_platform/backend/schema.prisma ./
RUN poetry run prisma generate

FROM debian:13-slim AS server_dependencies

WORKDIR /app

ENV POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/poetry/bin:$PATH

# Install Python and upgrade pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && python3 -m pip install --upgrade pip setuptools --break-system-packages

# Copy only necessary files from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3* /usr/local/lib/python3*
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry
# Copy Prisma binaries
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries


ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /app/autogpt_platform/autogpt_libs
RUN mkdir -p /app/autogpt_platform/backend

COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs

COPY autogpt_platform/backend/poetry.lock autogpt_platform/backend/pyproject.toml /app/autogpt_platform/backend/

WORKDIR /app/autogpt_platform/backend

FROM server_dependencies AS migrate

# Migration stage only needs schema and migrations - much lighter than full backend
COPY autogpt_platform/backend/schema.prisma /app/autogpt_platform/backend/
COPY autogpt_platform/backend/migrations /app/autogpt_platform/backend/migrations

FROM server_dependencies AS server

COPY autogpt_platform/backend /app/autogpt_platform/backend
RUN poetry install --no-ansi --only-root

ENV PORT=8000

CMD ["poetry", "run", "rest"]
