# Playbook Client

Production-ready Playbook Client implementation.

## Setup

1. **Credentials**:
   - Place `playbook/credentials.py` in the root of the repo (or `playbook_repo/playbook/credentials.py`).
   - Use `playbook/credentials.example.py` as a template.
   - Alternatively, use `.env` file or environment variables.

2. **Run Local**:
   ```bash
   ./run_local.sh
   ```
   Server will start on `http://localhost:8000`.

3. **Docker**:
   ```bash
   docker-compose up --build
   ```

## Testing

Run tests with `pytest`:
```bash
pytest tests/test_flow.py
```

If the Brain URL is invalid, tests will fail and create `KEY_VALIDATION_BLOCKER.md`.

## Features

- **Enterprise Mode**: File upload, extraction, training (mapping), execution.
- **Academic Mode**: Placeholder for future features.
- **Opportunities**: Displays opportunities from Brain.
- **Robust Error Handling**: Handles schema mismatches and connection errors gracefully (creating blocker files for critical failures).

## Deliverables

- `playbook_repo/` contains all source code.
- `playbook_release_v1.0.zip` will be generated on completion.
