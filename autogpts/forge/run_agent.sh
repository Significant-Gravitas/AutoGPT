kill $(lsof -t -i :8000)
poetry install
cp .env.example .env
pip uninstall agbenchmark
pip install -e ../../benchmark
poetry run python -m forge &
