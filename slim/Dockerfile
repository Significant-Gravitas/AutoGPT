#FROM ai_ticket
ARG OFFICIAL_PYTHON_IMAGE
FROM ${OFFICIAL_PYTHON_IMAGE} 

# Install browsers
# Install utilities
add requirements.txt  requirements.txt 
RUN pip install -r requirements.txt

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

ADD autogpts/autogpt /app/

ADD ./benchmark /benchmark/

WORKDIR /app
#RUN poetry run pip install /opt/ai-ticket
# RUN poetry install

RUN poetry install --verbose ||echo fail
RUN poetry install --verbose ||echo fail2

ENTRYPOINT ["poetry", "run", "autogpt", "--install-plugin-deps"]
#ENTRYPOINT ["poetry", "install" ]
CMD []
