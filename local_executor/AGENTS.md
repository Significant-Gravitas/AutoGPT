# Local Executor

This file guides coding agents working in `local_executor/`. It holds `autogpt-shim`, the companion people install on their own computers so Otto can work in a folder there (the Local PC execution target). It is a standalone Python package, not part of `autogpt_platform/`. It has its own tooling and is under the MIT license, like everything outside `autogpt_platform/` (see the root `LICENSE`).

## Tooling

- Python 3.11+, packaged with hatchling from `pyproject.toml`. It is not a Poetry project, so the platform's `poetry run` commands don't apply here.
- Set up: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- Test: `.venv/bin/python -m pytest tests/`. CI (`.github/workflows/local-executor-ci.yml`) runs the tests on Linux, macOS and Windows with Python 3.11 and 3.12.
- Lint and format with Ruff: `ruff check .` and `ruff format .`. Black does not run here. Types: `mypy autogpt_local_executor`.

## How it fits with the platform

- The wire protocol is `docs/PROTOCOL.md`. The platform's side lives in `autogpt_platform/backend/backend/copilot/tools/local_pc_*` and `autogpt_platform/backend/backend/api/features/local_executor/`. A protocol change updates both sides in the same pull request. Protocol 2.0 is the one exception: its platform side is pull request #13050, which lands first, and no platform speaking 1.x was ever released.
- Sign-in is OAuth with PKCE for the `CONNECT_LOCAL_EXECUTOR` scope (`docs/OAUTH_FLOW.md`). Each computer signs in on its own.
- The platform's setup card installs a pinned revision (`LocalExecutorSetup.tsx`). A change only reaches users once that pin moves.

## Rules

- It runs on people's computers with their permissions. Read `docs/SECURITY.md` before changing anything that touches files, commands, credentials or the network.
- Keep it working on Windows, macOS and Linux (`docs/CROSS_PLATFORM.md`).
- Text that reaches the platform UI or Otto follows the root `AGENTS.md` "User-facing copy" rules.
