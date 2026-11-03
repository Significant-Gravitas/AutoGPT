#!/usr/bin/env bash
set -euo pipefail

: "${POETRY_VERSION:?POETRY_VERSION must be selected from the tested lockfiles}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${UV_TOOL_BIN_DIR:?An isolated tool bin directory is required}"
[[ "${GITHUB_ACTIONS:-}" == true && "$(uname -s)" == Linux ]]

python_bin="$(command -v python)"
venv="$(pwd)/.venv"
if [[ -e "$venv" ]]; then
  echo "Expected a fresh backend environment: $venv" >&2
  exit 1
fi

uv tool install --python "$python_bin" --from "poetry==$POETRY_VERSION" \
  --with poetry-plugin-export==1.10.1 poetry
poetry_bin="$UV_TOOL_BIN_DIR/poetry"
requirements="$RUNNER_TEMP/backend-ci-requirements.txt"
"$poetry_bin" check --lock
"$poetry_bin" export --with dev --format requirements.txt --output "$requirements"
uv venv --python "$python_bin" "$venv"
uv pip sync --python "$venv/bin/python" "$requirements"

export VIRTUAL_ENV="$venv"
if [[ "$("$poetry_bin" env info --path)" != "$venv" ]]; then
  echo 'Poetry selected the wrong environment' >&2
  exit 1
fi
"$poetry_bin" install --only-root --no-interaction
uv pip check --python "$venv/bin/python"
git diff --exit-code -- poetry.lock pyproject.toml
printf '%s\n' "$UV_TOOL_BIN_DIR" >> "$GITHUB_PATH"
printf 'VIRTUAL_ENV=%s\n' "$venv" >> "$GITHUB_ENV"
