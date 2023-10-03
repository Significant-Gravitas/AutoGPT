ARG OFFICIAL_PYTHON_IMAGE
FROM ${OFFICIAL_PYTHON_IMAGE} 


# Install browsers
RUN apt-get update && apt-get install -y \
    chromium-driver firefox-esr ca-certificates gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install utilities
RUN apt-get update && apt-get install -y \
    curl jq wget git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
add requirements.txt  requirements.txt 
RUN pip install -r requirements.txt

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_NO_INTERACTION=1

ADD autogpts/autogpt /app/

ADD ./benchmark /benchmark/

WORKDIR /app
RUN poetry run pip install /opt/ai-ticket
# RUN poetry install

#RUN poetry install --verbose ||echo some failed lets inspect
#RUN poetry install --verbose ||echo some failed lets inspect

ENTRYPOINT ["poetry", "run", "autogpt", "--install-plugin-deps"]
#ENTRYPOINT ["poetry", "install" ]
CMD []
