ARG GOTTY_VERSION=v1.5.0

FROM debian:bullseye-slim AS builder

ARG GOTTY_VERSION

WORKDIR /build

ADD https://github.com/sorenisanerd/gotty/releases/download/${GOTTY_VERSION}/gotty_${GOTTY_VERSION}_linux_arm64.tar.gz gotty-aarch64.tar.gz
ADD https://github.com/sorenisanerd/gotty/releases/download/${GOTTY_VERSION}/gotty_${GOTTY_VERSION}_linux_amd64.tar.gz gotty-x86_64.tar.gz

RUN tar -xzvf "gotty-$(uname -m).tar.gz"
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/Torantulino/Auto-GPT.git

##############################

FROM python:3.11-slim

COPY --chmod=+x --from=builder /build/gotty /bin/gotty

WORKDIR /app

COPY --from=builder /build/Auto-GPT/ /app

RUN apt-get update && apt-get install git vim curl pkg-config \
libcairo2-dev build-essential libgirepository1.0-dev \
gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base \
gir1.2-gst-plugins-base-1.0 sudo wget curl -y

ARG UID=1000
ARG GID=1000
RUN adduser --disabled-password --uid 1000 --home /app --gecos "" auto-gpt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install pycairo PyGObject && \
    chown -R 1000:1000 /app 

USER 1000

ENV PINECONE_API_KEY=${PINECONE_API_KEY} \
    PINECONE_ENV=${PINECONE_ENV} \
    OPENAI_API_KEY=${OPENAI_API_KEY} \
    ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY} \
    ELEVENLABS_VOICE_1_ID=${ELEVENLABS_VOICE_1_ID} \
    ELEVENLABS_VOICE_2_ID=${ELEVENLABS_VOICE_2_ID} \
    SMART_LLM_MODEL=${SMART_LLM_MODEL} \
    FAST_LLM_MODEL=${FAST_LLM_MODEL} \
    GOOGLE_API_KEY=${GOOGLE_API_KEY} \
    CUSTOM_SEARCH_ENGINE_ID=${CUSTOM_SEARCH_ENGINE_ID} \
    USE_AZURE=${USE_AZURE} \
    OPENAI_AZURE_API_BASE=${OPENAI_AZURE_API_BASE} \
    OPENAI_AZURE_API_VERSION=${OPENAI_AZURE_API_VERSION} \
    OPENAI_AZURE_DEPLOYMENT_ID=${OPENAI_AZURE_DEPLOYMENT_ID} \
    IMAGE_PROVIDER=${IMAGE_PROVIDER} \
    HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
    USE_MAC_OS_TTS=${USE_MAC_OS_TTS} 

EXPOSE 8080

WORKDIR /app

