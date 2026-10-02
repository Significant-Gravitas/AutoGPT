# Build context: autogpt_platform/swap_proxy
FROM python:3.13-slim AS builder

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

# Same Poetry version as the lockfile was written with.
RUN pip install "poetry==2.2.1"

WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root
COPY swap_proxy ./swap_proxy
RUN poetry install --only main


FROM python:3.13-slim

RUN useradd --system --create-home --home-dir /var/lib/swap-proxy swap-proxy
COPY --from=builder /app /app

# The signing CA is not in the image and is not generated: mount the provisioned
# one (mitmproxy-ca.pem, plus mitmproxy-dhparam.pem if the mount is read-only)
# at SWAP_PROXY_CONFDIR.  Without it the service refuses to start.  See README.
USER swap-proxy
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SWAP_PROXY_CONFDIR=/var/lib/swap-proxy/ca

EXPOSE 1080
CMD ["swap-proxy"]
