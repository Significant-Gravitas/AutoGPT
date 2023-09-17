#!/bin/bash


kill $(lsof -t -i :8000)
poetry install
poetry run pip3 uninstall agbenchmark --yes
poetry run pip3 install -e ../../benchmark
poetry run python3 -m forge &
export PYTHONPATH=$PYTHONPATH:../../benchmark/agbenchmark
poetry run python3 -m agbenchmark "$@"
