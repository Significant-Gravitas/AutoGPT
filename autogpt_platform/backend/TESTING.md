# Backend Testing Guide

This guide covers testing practices for the AutoGPT Platform backend, with a focus on snapshot testing for API endpoints.

## Table of Contents
- [Overview](#overview)
- [Running Tests](#running-tests)
- [Snapshot Testing](#snapshot-testing)
- [Writing Tests for API Routes](#writing-tests-for-api-routes)
- [Best Practices](#best-practices)

## Overview

The backend uses pytest for testing with the following key libraries:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-mock` - Mocking support
- `pytest-snapshot` - Snapshot testing for API responses

## Running Tests

### Run all tests
```bash
poetry run test
```

### Run specific test file
```bash
poetry run pytest path/to/test_file.py
```

### Run with verbose output
```bash
poetry run pytest -v
```

### Run with coverage
```bash
poetry run pytest --cov=backend
```

## Snapshot Testing

Snapshot testing captures the output of your code and compares it against previously saved snapshots. This is particularly useful for testing API responses.

### How Snapshot Testing Works

1. First run: Creates snapshot files in `snapshots/` directories
2. Subsequent runs: Compares output against saved snapshots
3. Changes detected: Test fails if output differs from snapshot

### Creating/Updating Snapshots

When you first write a test or when the expected output changes:

```bash
poetry run pytest path/to/test.py --snapshot-update
```

⚠️ **Important**: Always review snapshot changes before committing! Use `git diff` to verify the changes are expected.

### Snapshot Test Example

```python
import json
from pytest_snapshot.plugin import Snapshot

def test_api_endpoint(snapshot: Snapshot):
    response = client.get("/api/endpoint")
    
    # Snapshot the response
    snapshot.snapshot_dir = "snapshots"
snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "endpoint_response"
    )
```

### Best Practices for Snapshots

1. **Use descriptive names**: `"user_list_response"` not `"response1"`
2. **Sort JSON keys**: Ensures consistent snapshots
3. **Format JSON**: Use `indent=2` for readable diffs
4. **Exclude dynamic data**: Remove timestamps, IDs, etc. that change between runs

Example of excluding dynamic data:
```python
response_data = response.json()
# Remove dynamic fields for snapshot
response_data.pop("created_at", None)
response_data.pop("id", None)

snapshot.snapshot_dir = "snapshots"
snapshot.assert_match(
    json.dumps(response_data, indent=2, sort_keys=True),
    "static_response_data"
)
```

## Writing Tests for API Routes

### Basic Structure

```python
import json
import fastapi
import fastapi.testclient
import pytest
from pytest_snapshot.plugin import Snapshot

from backend.server.v2.myroute import router

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

def test_endpoint_success(snapshot: Snapshot):
    response = client.get("/endpoint")
    assert response.status_code == 200
    
    # Test specific fields
    data = response.json()
    assert data["status"] == "success"
    
    # Snapshot the full response
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True),
        "endpoint_success_response"
    )
```

### Testing with Authentication

```python
def override_auth_middleware():
    return {"sub": "test-user-id"}

def override_get_user_id():
    return "test-user-id"

app.dependency_overrides[auth_middleware] = override_auth_middleware
app.dependency_overrides[get_user_id] = override_get_user_id
```

### Mocking External Services

```python
def test_external_api_call(mocker, snapshot):
    # Mock external service
    mock_response = {"external": "data"}
    mocker.patch(
        "backend.services.external_api.call",
        return_value=mock_response
    )
    
    response = client.post("/api/process")
    assert response.status_code == 200
    
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "process_with_external_response"
    )
```

## Best Practices

### 1. Test Organization
- Place tests next to the code: `routes.py` → `routes_test.py`
- Use descriptive test names: `test_create_user_with_invalid_email`
- Group related tests in classes when appropriate

### 2. Test Coverage
- Test happy path and error cases
- Test edge cases (empty data, invalid formats)
- Test authentication and authorization

### 3. Snapshot Testing Guidelines
- Review all snapshot changes carefully
- Don't snapshot sensitive data
- Keep snapshots focused and minimal
- Update snapshots intentionally, not accidentally

### 4. Async Testing
- Use regular `def` for FastAPI TestClient tests
- Use `async def` with `@pytest.mark.asyncio` for testing async functions directly

### 5. Fixtures
Create reusable fixtures for common test data:

```python
@pytest.fixture
def sample_user():
    return {
        "email": "test@example.com",
        "name": "Test User"
    }

def test_create_user(sample_user, snapshot):
    response = client.post("/users", json=sample_user)
    # ... test implementation
```

## CI/CD Integration

The GitHub Actions workflow automatically runs tests on:
- Pull requests
- Pushes to main branch

Snapshot tests work in CI by:
1. Committing snapshot files to the repository
2. CI compares against committed snapshots
3. Fails if snapshots don't match

## Troubleshooting

### Snapshot Mismatches
- Review the diff carefully
- If changes are expected: `poetry run pytest --snapshot-update`
- If changes are unexpected: Fix the code causing the difference

### Async Test Issues
- Ensure async functions use `@pytest.mark.asyncio`
- Use `AsyncMock` for mocking async functions
- FastAPI TestClient handles async automatically

### Import Errors
- Check that all dependencies are in `pyproject.toml`
- Run `poetry install` to ensure dependencies are installed
- Verify import paths are correct

## Summary

Snapshot testing provides a powerful way to ensure API responses remain consistent. Combined with traditional assertions, it creates a robust test suite that catches regressions while remaining maintainable.

Remember: Good tests are as important as good code!