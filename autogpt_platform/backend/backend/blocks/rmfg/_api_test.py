"""Unit tests for the RMFG client's request shaping and error handling."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.blocks.rmfg import _http
from backend.blocks.rmfg._api import SERVER_WAIT_SECONDS, RMFGClient
from backend.blocks.rmfg._http import POLL_INITIAL_SECONDS, RMFGError, parse_body
from backend.blocks.rmfg._models import ResourceError
from backend.blocks.rmfg._testdata import TEST_DESIGN, TEST_PENDING_DESIGN, TEST_SHIP_TO
from backend.blocks.rmfg._types import (
    DesignStatus,
    HardwareKind,
    ManufacturingConfiguration,
    PaymentType,
    Process,
    QuoteItemRequest,
)
from backend.data.model import APIKeyCredentials


def _credentials() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="01234567-89ab-cdef-0123-456789abcdef",
        provider="rmfg",
        api_key=SecretStr("test-key"),
        title="Test key",
        expires_at=None,
    )


class _FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, body: str = ""):
        self.status = status
        self._payload = payload
        self._body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload

    def text(self) -> str:
        if self._body:
            return self._body
        return "{}" if self._payload is not None else ""


def _client_with(*responses: _FakeResponse) -> tuple[RMFGClient, AsyncMock]:
    """Build a client whose HTTP layer is swapped for a recording mock."""
    client = RMFGClient(_credentials())
    request = AsyncMock(side_effect=list(responses))
    # Patch the transport itself so the tests cover the client's own request
    # shaping (params, body, headers) without any network access.
    requests: Any = client.requests
    requests.request = request
    return client, request


def _sent(request: AsyncMock, call: int = 0) -> tuple[str, str, dict[str, Any]]:
    method, url = request.await_args_list[call].args
    return method, url, request.await_args_list[call].kwargs


class TestAuth:
    def test_bearer_header_is_set_once_for_every_call(self):
        client = RMFGClient(_credentials())
        assert client.requests.extra_headers == {
            "Authorization": "Bearer test-key",
            "Accept": "application/json",
        }
        assert client.requests.raise_for_status is False


class TestParseBody:
    def test_returns_object_on_success(self):
        assert parse_body(_FakeResponse(200, {"id": "x"})) == {"id": "x"}

    def test_empty_success_body_is_an_empty_object(self):
        assert parse_body(_FakeResponse(204)) == {}

    def test_error_envelope_is_rendered(self):
        response = _FakeResponse(
            422,
            {
                "error": {
                    "type": "invalid_request_error",
                    "code": "validation_error",
                    "message": "quantity must be positive",
                    "param": "items.0.quantity",
                    "request_id": "req_1",
                }
            },
        )
        with pytest.raises(RMFGError) as exc_info:
            parse_body(response)
        assert exc_info.value.status == 422
        assert exc_info.value.code == "validation_error"
        assert "quantity must be positive (field: items.0.quantity)" in str(
            exc_info.value
        )
        assert "req_1" in str(exc_info.value)

    def test_auth_failures_point_at_the_key(self):
        response = _FakeResponse(
            401, {"error": {"type": "authentication_error", "message": "bad key"}}
        )
        with pytest.raises(RMFGError, match="Check the RMFG API key"):
            parse_body(response)

    def test_non_json_failure_keeps_the_status(self):
        with pytest.raises(RMFGError) as exc_info:
            parse_body(_FakeResponse(502, body="<html>Bad Gateway</html>"))
        assert exc_info.value.status == 502
        assert "Bad Gateway" in str(exc_info.value)

    def test_non_json_success_is_still_an_error(self):
        with pytest.raises(RMFGError, match="not JSON"):
            parse_body(_FakeResponse(200, body="ok"))


class TestPagination:
    async def test_follows_next_cursor_until_has_more_is_false(self):
        client, request = _client_with(
            _FakeResponse(
                200, {"data": [{"id": "a"}], "has_more": True, "next_cursor": "c2"}
            ),
            _FakeResponse(200, {"data": [{"id": "b"}], "has_more": False}),
        )

        materials = await client.list_materials()

        assert [m.id for m in materials] == ["a", "b"]
        assert request.await_count == 2
        _, _, first = _sent(request, 0)
        _, _, second = _sent(request, 1)
        assert "cursor" not in first["params"]
        assert second["params"]["cursor"] == "c2"
        assert second["params"]["limit"] == 500

    async def test_finish_process_filter_is_a_query_param(self):
        client, request = _client_with(_FakeResponse(200, {"data": []}))

        await client.list_finishes(Process.TUBE_LASER)

        _, url, kwargs = _sent(request)
        assert url.endswith("/v1/finishes")
        assert kwargs["params"]["process"] == "tube_laser"

    async def test_hardware_kind_selects_the_path(self):
        client, request = _client_with(_FakeResponse(200, {"data": []}))

        await client.list_hardware(HardwareKind.STANDOFFS)

        _, url, _ = _sent(request)
        assert url.endswith("/v1/hardware/standoffs")


class TestAnalyze:
    async def test_uploads_multipart_with_idempotency_and_wait(self):
        client, request = _client_with(
            _FakeResponse(202, TEST_PENDING_DESIGN.model_dump(mode="json"))
        )

        design = await client.analyze("bracket.step", b"ISO-10303-21;", "key-1")

        assert design.status == DesignStatus.PROCESSING
        method, url, kwargs = _sent(request)
        assert (method, url) == ("POST", "https://api.rmfg.com/v1/analyze")
        assert kwargs["headers"]["Idempotency-Key"] == "key-1"
        assert kwargs["headers"]["Prefer"] == f"wait={SERVER_WAIT_SECONDS}"
        assert kwargs["json"] is None
        field, (filename, handle, content_type) = kwargs["files"][0]
        assert (field, filename, content_type) == (
            "file",
            "bracket.step",
            "application/step",
        )
        assert handle.getvalue() == b"ISO-10303-21;"

    async def test_no_idempotency_header_when_key_is_empty(self):
        client, request = _client_with(
            _FakeResponse(200, TEST_DESIGN.model_dump(mode="json"))
        )

        await client.get_design(TEST_DESIGN.id)

        _, _, kwargs = _sent(request)
        assert "Idempotency-Key" not in kwargs["headers"]
        assert "Prefer" not in kwargs["headers"]


class TestPolling:
    async def test_returns_immediately_when_already_ready(self):
        client, request = _client_with()

        design = await client.wait_for_design(TEST_DESIGN, timeout_seconds=5)

        assert design is TEST_DESIGN
        assert request.await_count == 0

    async def test_polls_until_ready(self, monkeypatch: pytest.MonkeyPatch):
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(_http, "asyncio", SimpleNamespace(sleep=fake_sleep))
        client, request = _client_with(
            _FakeResponse(200, TEST_PENDING_DESIGN.model_dump(mode="json")),
            _FakeResponse(200, TEST_DESIGN.model_dump(mode="json")),
        )

        design = await client.wait_for_design(TEST_PENDING_DESIGN, timeout_seconds=60)

        assert design.status == DesignStatus.READY
        assert request.await_count == 2
        assert sleeps[0] == POLL_INITIAL_SECONDS
        assert sleeps[1] > sleeps[0], "backoff grows between polls"

    async def test_failed_analysis_raises_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            _http, "asyncio", SimpleNamespace(sleep=AsyncMock(return_value=None))
        )
        failed = TEST_DESIGN.model_copy(
            update={
                "status": DesignStatus.FAILED,
                "error": ResourceError(code="bad_geometry", message="Not a solid body"),
            }
        )
        client, _ = _client_with(_FakeResponse(200, failed.model_dump(mode="json")))

        with pytest.raises(RMFGError, match="Not a solid body"):
            await client.wait_for_design(TEST_PENDING_DESIGN, timeout_seconds=60)

    async def test_times_out_instead_of_polling_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            _http, "asyncio", SimpleNamespace(sleep=AsyncMock(return_value=None))
        )
        # Two reads before the deadline, then the clock jumps past it. Scoped
        # to the module so the event loop's own clock is untouched.
        ticks = [0.0, 0.0, 100.0]
        monkeypatch.setattr(
            _http,
            "time",
            SimpleNamespace(
                monotonic=lambda: ticks.pop(0) if len(ticks) > 1 else ticks[0]
            ),
        )
        client, _ = _client_with(
            _FakeResponse(200, TEST_PENDING_DESIGN.model_dump(mode="json"))
        )

        with pytest.raises(TimeoutError, match=TEST_PENDING_DESIGN.id):
            await client.wait_for_design(TEST_PENDING_DESIGN, timeout_seconds=10)


class TestBaskets:
    async def test_quote_body_matches_the_documented_shape(self):
        client, request = _client_with(
            _FakeResponse(200, {"id": "qte_1", "status": "processing"})
        )
        configuration = ManufacturingConfiguration.model_validate(
            {"parts": [{"part_id": "prt_1", "material_id": "mat_1"}]}
        )
        item = QuoteItemRequest(
            design_id="dsn_1", quantity=10, configuration=configuration
        )

        await client.create_quote([item], idempotency_key="q-1")

        _, url, kwargs = _sent(request)
        assert url.endswith("/v1/quotes")
        assert kwargs["json"] == {
            "items": [
                {
                    "design_id": "dsn_1",
                    "quantity": 10,
                    "quantity_options": [],
                    "configuration": {
                        "defaults": {},
                        "parts": [
                            {
                                "part_id": "prt_1",
                                "material_id": "mat_1",
                                "taps": [],
                                "studs": [],
                                "nuts": [],
                                "standoffs": [],
                                "countersinks": [],
                            }
                        ],
                        "assembly_operations": [],
                        "accepted_risks": [],
                    },
                }
            ]
        }
        assert kwargs["headers"]["Prefer"] == f"wait={SERVER_WAIT_SECONDS}"

    async def test_cart_includes_destination_and_shipping_choice(self):
        client, request = _client_with(_FakeResponse(201, {"id": "crt_1"}))

        await client.create_cart(
            [QuoteItemRequest(design_id="dsn_1")],
            ship_to=TEST_SHIP_TO,
            shipping_option_id="ship_1",
            client_reference_id="po-7",
        )

        _, _, kwargs = _sent(request)
        body = kwargs["json"]
        assert body["ship_to"]["postal_code"] == "78701"
        assert "street2" not in body["ship_to"], "unset optionals are omitted"
        assert body["shipping_option_id"] == "ship_1"
        assert body["client_reference_id"] == "po-7"

    async def test_update_only_sends_what_changed(self):
        client, request = _client_with(_FakeResponse(200, {"id": "crt_1"}))

        await client.update_cart("crt_1", shipping_option_id="ship_2")

        method, url, kwargs = _sent(request)
        assert (method, url) == ("PATCH", "https://api.rmfg.com/v1/carts/crt_1")
        assert kwargs["json"] == {"shipping_option_id": "ship_2"}


class TestPayment:
    async def test_card_on_file_is_the_documented_body(self):
        client, request = _client_with(_FakeResponse(200, {"id": "crt_1"}))

        await client.pay_cart("crt_1", idempotency_key="pay-1")

        method, url, kwargs = _sent(request)
        assert (method, url) == ("POST", "https://api.rmfg.com/v1/carts/crt_1/pay")
        assert kwargs["json"] == {"payment": {"type": "card_on_file"}}
        assert kwargs["headers"]["Idempotency-Key"] == "pay-1"

    async def test_payment_method_carries_its_id_and_contact(self):
        client, request = _client_with(_FakeResponse(200, {"id": "crt_1"}))

        await client.pay_cart(
            "crt_1",
            payment_type=PaymentType.PAYMENT_METHOD,
            payment_method_id="pm_123",
            customer_email="ada@example.com",
        )

        _, _, kwargs = _sent(request)
        assert kwargs["json"] == {
            "payment": {"type": "payment_method", "id": "pm_123"},
            "customer_email": "ada@example.com",
        }


class TestOrders:
    async def test_list_returns_cursor_only_when_more_pages_exist(self):
        client, _ = _client_with(
            _FakeResponse(
                200,
                {"data": [{"id": "ord_1"}], "has_more": True, "next_cursor": "n"},
            )
        )
        orders, cursor = await client.list_orders(limit=1)
        assert [o.id for o in orders] == ["ord_1"]
        assert cursor == "n"

    async def test_last_page_has_no_cursor(self):
        client, _ = _client_with(
            _FakeResponse(200, {"data": [], "has_more": False, "next_cursor": "n"})
        )
        _, cursor = await client.list_orders()
        assert cursor is None
