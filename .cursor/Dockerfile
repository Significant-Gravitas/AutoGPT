FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        fuse-overlayfs \
        iptables \
        build-essential \
        python3 \
        python3-pip \
        python3-venv \
        make \
        lsb-release \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/docker \
    && printf '{\n  "storage-driver": "fuse-overlayfs"\n}\n' > /etc/docker/daemon.json

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

ENV NVM_DIR=/opt/nvm
RUN mkdir -p "$NVM_DIR" \
    && curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | NVM_DIR="$NVM_DIR" PROFILE=/dev/null bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install 24 \
    && nvm alias default 24 \
    && corepack enable \
    && corepack prepare pnpm@10.33.3 --activate

ENV PATH="/opt/nvm/versions/node/v24.19.0/bin:/root/.local/bin:${PATH}"
