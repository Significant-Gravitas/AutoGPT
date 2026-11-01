"""What an error response tells the caller."""

import fastapi
import fastapi.testclient
import pydantic
import pytest

from backend.api.utils.exceptions import add_exception_handlers


class _Items(pydantic.BaseModel):
    items: list[int] = pydantic.Field(max_length=3)


def _client(*, server_error_detail: bool) -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("Client is not connected to the query engine")

    @app.get("/bad")
    async def bad() -> None:
        raise ValueError("limit must be positive")

    @app.post("/items")
    async def items(body: _Items) -> None:
        return None

    add_exception_handlers(app, server_error_detail=server_error_detail)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("server_error_detail", [False, True])
def test_a_server_error_names_its_cause_only_where_asked(
    server_error_detail: bool,
) -> None:
    """Third parties call v1, and the text can name internal services."""
    response = _client(server_error_detail=server_error_detail).get("/boom")

    assert response.status_code == 500
    assert ("query engine" in response.json()["detail"]) is server_error_detail


def test_a_client_error_still_says_what_was_wrong() -> None:
    response = _client(server_error_detail=False).get("/bad")

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be positive"


def test_a_validation_error_does_not_echo_or_log_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = 987654321
    response = _client(server_error_detail=False).post(
        "/items", json={"items": [marker] * 50}
    )

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "items"]
    assert error["type"] == "too_long"
    assert "input" not in error
    assert str(marker) not in response.text
    assert str(marker) not in caplog.text
