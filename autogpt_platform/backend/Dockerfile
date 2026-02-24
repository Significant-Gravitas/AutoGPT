# ============================ DEPENDENCY BUILDER ============================ #

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
COPY autogpt_platform/backend/backend/data/partial_types.py ./backend/data/partial_types.py
COPY autogpt_platform/backend/gen_prisma_types_stub.py ./
RUN poetry run prisma generate && poetry run gen-prisma-stub

# =============================== DB MIGRATOR =============================== #

# Lightweight migrate stage - only needs Prisma CLI, not full Python environment
FROM debian:13-slim AS migrate

WORKDIR /app/autogpt_platform/backend

ENV DEBIAN_FRONTEND=noninteractive

# Install only what's needed for prisma migrate: Node.js and minimal Python for prisma-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js from builder (needed for Prisma CLI)
COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules
COPY --from=builder /usr/bin/npm /usr/bin/npm

# Copy Prisma binaries
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries

# Install prisma-client-py directly (much smaller than copying full venv)
RUN pip3 install prisma>=0.15.0 --break-system-packages

COPY autogpt_platform/backend/schema.prisma ./
COPY autogpt_platform/backend/backend/data/partial_types.py ./backend/data/partial_types.py
COPY autogpt_platform/backend/gen_prisma_types_stub.py ./
COPY autogpt_platform/backend/migrations ./migrations

# ============================== BACKEND SERVER ============================== #

FROM debian:13-slim AS server

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, FFmpeg, ImageMagick, and CLI tools for agent use.
# bubblewrap provides OS-level sandbox (whitelist-only FS + no network)
# for the bash_exec MCP tool.
# Using --no-install-recommends saves ~650MB by skipping unnecessary deps like llvm, mesa, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3-pip \
    ffmpeg \
    imagemagick \
    jq \
    ripgrep \
    tree \
    bubblewrap \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry (build-time only, for `poetry install --only-root` to create entry points)
COPY --from=builder /usr/local/lib/python3* /usr/local/lib/python3*
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry
# Copy Node.js installation for Prisma
COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules
COPY --from=builder /usr/bin/npm /usr/bin/npm
COPY --from=builder /usr/bin/npx /usr/bin/npx
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries

WORKDIR /app/autogpt_platform/backend

# Copy only the .venv from builder (not the entire /app directory)
# The .venv includes the generated Prisma client
COPY --from=builder /app/autogpt_platform/backend/.venv ./.venv
ENV PATH="/app/autogpt_platform/backend/.venv/bin:$PATH"

# Copy dependency files + autogpt_libs (path dependency)
COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs
COPY autogpt_platform/backend/poetry.lock autogpt_platform/backend/pyproject.toml ./

# Copy backend code + docs (for Copilot docs search)
COPY autogpt_platform/backend ./
COPY docs /app/docs
# Install the project package to create entry point scripts in .venv/bin/
# (e.g., rest, executor, ws, db, scheduler, notification - see [tool.poetry.scripts])
RUN POETRY_VIRTUALENVS_CREATE=true POETRY_VIRTUALENVS_IN_PROJECT=true \
    poetry install --no-ansi --only-root

ENV PORT=8000

CMD ["rest"]
