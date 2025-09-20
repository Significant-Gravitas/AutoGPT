FROM debian:13-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN echo 'Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;\n' > /etc/apt/apt.conf.d/99fixbadproxy

# Install Node.js repository key and setup
RUN apt-get update --allow-releaseinfo-change --fix-missing \
    && apt-get install -y curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list

# Update package list and install Python, Node.js, and build dependencies
RUN apt-get update \
    && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    build-essential \
    libpq5 \
    libz-dev \
    libssl-dev \
    postgresql-client \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH=/opt/poetry/bin:$PATH

RUN pip3 install poetry --break-system-packages

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
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/poetry/bin:$PATH

# Install Python without upgrading system-managed packages
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3* /usr/local/lib/python3*
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry
# Copy Node.js installation for Prisma
COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules
COPY --from=builder /usr/bin/npm /usr/bin/npm
COPY --from=builder /usr/bin/npx /usr/bin/npx
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries

ENV PATH="/app/autogpt_platform/backend/.venv/bin:$PATH"

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
