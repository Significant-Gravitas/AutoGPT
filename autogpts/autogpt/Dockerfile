FROM ai_ticket
ARG BUILD_TYPE=dev
RUN apt-get update && apt-get install -y \
    chromium-driver firefox-esr ca-certificates gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    curl jq wget git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config installer.max-workers 10
ADD autogpts/autogpt /app/
WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY README.md poetry.lock ./
ADD ./benchmark /
RUN poetry add /opt/ai-ticket
RUN poetry install

ENTRYPOINT ["poetry", "run", "autogpt", "--install-plugin-deps"]
CMD []
