#!/bin/sh

# Necessary to prevent forge and agbenchmark from breaking each others' install:
# https://github.com/python-poetry/poetry/issues/6958
POETRY_INSTALLER_PARALLEL=false \
poetry install --no-interaction --extras benchmark

echo "Setup completed successfully."
