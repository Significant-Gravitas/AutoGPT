#!/bin/zsh
echo "Running black..."
poetry run black .

echo "Running isort..."
poetry run isort .

echo "Running autopep8..."
poetry run autopep8 --in-place --recursive  .  

echo "Running fix_w293.py..."
python3 .vscode/fix_w293.py

echo "Running flake8..."
poetry run flake8 .