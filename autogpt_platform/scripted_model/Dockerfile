# Build context: autogpt_platform/scripted_model
# Standard library only, so the image builds in seconds with nothing to install.
FROM python:3.13-slim

RUN useradd --system --no-create-home scripted-model
WORKDIR /app
COPY scripted_model ./scripted_model

USER scripted-model
ENV PYTHONUNBUFFERED=1 \
    SCRIPTED_MODEL_PORT=4010 \
    SCRIPTED_MODEL_SCRIPTS_DIR=/scripts

EXPOSE 4010
CMD ["python", "-m", "scripted_model"]
