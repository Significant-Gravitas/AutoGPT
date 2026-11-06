import prisma
import pytest
from prisma.engine import _http
from prisma.engine import utils as engine_utils
from prisma.errors import DataError, UniqueViolationError

from backend.data import db
from backend.data.prisma_compat import _PATCH_ATTR, patch_handle_response_errors


@pytest.fixture(scope="session")
def server():
    yield None


def _handle(payload: list) -> None:
    engine_utils.handle_response_errors(object(), payload)


def _error(*, code: str | None, message: str, meta) -> list:
    return [
        {
            "error": message,
            "user_facing_error": {
                "is_panic": False,
                "message": message,
                "meta": meta,
                "error_code": code,
            },
        }
    ]


def test_pinned_prisma_version_is_the_archived_release():
    assert prisma.__version__ == "0.15.0"
    assert db.prisma is not None
    assert getattr(engine_utils.handle_response_errors, _PATCH_ATTR, False)


def test_p1017_with_null_meta_raises_data_error():
    payload = _error(
        code="P1017",
        message="Server has closed the connection.",
        meta=None,
    )
    with pytest.raises(DataError, match="Server has closed the connection"):
        _handle(payload)


def test_null_user_facing_error_raises_data_error():
    payload = [{"error": "broken payload", "user_facing_error": None}]
    with pytest.raises(DataError):
        _handle(payload)


def test_p2002_with_null_meta_raises_unique_violation():
    payload = _error(
        code="P2002",
        message="Unique constraint failed on the fields: (`email`)",
        meta=None,
    )
    with pytest.raises(UniqueViolationError):
        _handle(payload)


def test_well_formed_p1001_still_raises_data_error():
    payload = _error(
        code="P1001",
        message="Can't reach database server at `localhost`:`5432`",
        meta={"database_host": "localhost", "database_port": 5432},
    )
    with pytest.raises(DataError, match="Can't reach database server"):
        _handle(payload)


def test_http_engine_p1017_null_meta_raises_data_error():
    engine = _http.BaseHTTPEngine(url="http://unused")
    with pytest.raises(DataError, match="Server has closed the connection"):
        engine._process_response_data(
            data={
                "errors": _error(
                    code="P1017",
                    message="Server has closed the connection.",
                    meta=None,
                )
            },
            response=object(),
        )


def test_patch_is_idempotent():
    first = engine_utils.handle_response_errors
    patch_handle_response_errors()
    patch_handle_response_errors()
    assert engine_utils.handle_response_errors is first
    assert _http.utils.handle_response_errors is first
    assert getattr(first, _PATCH_ATTR, False)
