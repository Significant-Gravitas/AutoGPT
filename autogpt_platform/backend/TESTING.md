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

from backend.api.features.myroute import router

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

For the main API routes that use JWT authentication, auth is provided by the `autogpt_libs.auth` module. If the test actually uses the `user_id`, the recommended approach for testing is to mock the `get_jwt_payload` function, which underpins all higher-level auth functions used in the API (`requires_user`, `requires_admin_user`, `get_user_id`).

If the test doesn't need the `user_id` specifically, mocking is not necessary as during tests auth is disabled anyway (see `conftest.py`).

#### Using Global Auth Fixtures

Two global auth fixtures are provided by `backend/server/conftest.py`:

- `mock_jwt_user` - Regular user with `test_user_id` ("test-user-id")
- `mock_jwt_admin` - Admin user with `admin_user_id` ("admin-user-id")

These provide the easiest way to set up authentication mocking in test modules:

```python
import fastapi
import fastapi.testclient
import pytest
from backend.api.features.myroute import router

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user['get_jwt_payload']
    yield
    app.dependency_overrides.clear()
```

For admin-only endpoints, use `mock_jwt_admin` instead:

```python
@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_admin):
    """Setup auth overrides for admin tests"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin['get_jwt_payload']
    yield
    app.dependency_overrides.clear()
```

The IDs are also available separately as fixtures:

- `test_user_id`
- `admin_user_id`
- `target_user_id` (for admin <-> user operations)

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

#### Global Fixtures (conftest.py)

Authentication fixtures are available globally from `conftest.py`:

- `mock_jwt_user` - Standard user authentication
- `mock_jwt_admin` - Admin user authentication
- `configured_snapshot` - Pre-configured snapshot fixture

#### Custom Fixtures

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

#### Test Isolation

All tests must use fixtures that ensure proper isolation:

- Authentication overrides are automatically cleaned up after each test
- Database connections are properly managed with cleanup
- Mock objects are reset between tests

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
