"""Async client for the RMFG Manufacturing API.

Responses are validated into the models in ``_models`` and
``_models_commerce`` so blocks never hand raw dicts to downstream nodes.
"""

from io import BytesIO
from typing import Any, Optional

from pydantic import TypeAdapter

from backend.sdk import APIKeyCredentials, Requests

from ._http import RMFGError, parse_body, poll
from ._models import (
    Design,
    DFMReport,
    Finish,
    HardwareOption,
    Material,
    PowderCoatColor,
    TubeProfile,
)
from ._models_commerce import Cart, Order, Quote, ReviewLink
from ._types import (
    DesignStatus,
    HardwareKind,
    ManufacturingConfiguration,
    PaymentType,
    Process,
    QuoteItemRequest,
    QuoteStatus,
    ShipTo,
)

RMFG_API_URL = "https://api.rmfg.com"

# How long a create call asks the server to hold the connection while the
# resource finishes, before the client falls back to polling.
SERVER_WAIT_SECONDS = 20

_JSON_LIST = TypeAdapter(list[dict[str, Any]])


class RMFGClient:
    """Thin wrapper over the RMFG Manufacturing API v1."""

    def __init__(self, credentials: APIKeyCredentials):
        self.base_url = f"{RMFG_API_URL}/v1"
        # 429/5xx are retried by ``Requests`` itself with jittered backoff; the
        # attempt cap keeps a stuck endpoint from consuming the whole block
        # timeout before the error surfaces.
        self.requests = Requests(
            raise_for_status=False,
            retry_max_attempts=5,
            extra_headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
                "Accept": "application/json",
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        files: Optional[list[tuple[str, tuple[str, BytesIO, str]]]] = None,
        idempotency_key: str = "",
        wait: bool = False,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if wait:
            headers["Prefer"] = f"wait={SERVER_WAIT_SECONDS}"
        response = await self.requests.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            params={k: v for k, v in (params or {}).items() if v not in (None, "")},
            json=body,
            files=files,
        )
        return parse_body(response)

    async def _list_all(self, path: str, params: Optional[dict] = None) -> list[dict]:
        """Follow ``next_cursor`` until ``has_more`` is false."""
        items: list[dict] = []
        cursor: Optional[str] = None
        while True:
            page = await self._request(
                "GET", path, params={**(params or {}), "cursor": cursor}
            )
            items.extend(_JSON_LIST.validate_python(page.get("data") or []))
            cursor = page.get("next_cursor")
            if not page.get("has_more") or not cursor:
                return items

    # --- catalog -----------------------------------------------------------

    async def list_materials(self) -> list[Material]:
        rows = await self._list_all("/materials", {"limit": 500})
        return [Material.model_validate(row) for row in rows]

    async def list_tube_profiles(self) -> list[TubeProfile]:
        rows = await self._list_all("/tube-profiles", {"limit": 500})
        return [TubeProfile.model_validate(row) for row in rows]

    async def list_finishes(self, process: Optional[Process] = None) -> list[Finish]:
        params = {"limit": 500, "process": process.value if process else None}
        rows = await self._list_all("/finishes", params)
        return [Finish.model_validate(row) for row in rows]

    async def list_powder_coat_colors(self) -> list[PowderCoatColor]:
        rows = await self._list_all("/powder-coat-colors", {"limit": 500})
        return [PowderCoatColor.model_validate(row) for row in rows]

    async def list_hardware(self, kind: HardwareKind) -> list[HardwareOption]:
        rows = await self._list_all(f"/hardware/{kind.value}", {"limit": 500})
        return [HardwareOption.model_validate(row) for row in rows]

    # --- designs -----------------------------------------------------------

    async def analyze(
        self, filename: str, content: bytes, idempotency_key: str = ""
    ) -> Design:
        data = await self._request(
            "POST",
            "/analyze",
            files=[("file", (filename, BytesIO(content), "application/step"))],
            idempotency_key=idempotency_key,
            wait=True,
        )
        return Design.model_validate(data)

    async def get_design(self, design_id: str) -> Design:
        return Design.model_validate(
            await self._request("GET", f"/designs/{design_id}")
        )

    async def wait_for_design(self, design: Design, timeout_seconds: float) -> Design:
        """Poll until analysis leaves ``queued``/``processing``."""
        pending = {DesignStatus.QUEUED, DesignStatus.PROCESSING}
        design = await poll(
            lambda: self.get_design(design.id),
            initial=design,
            is_pending=lambda d: d.status in pending,
            timeout_seconds=timeout_seconds,
            what=f"design {design.id}",
        )
        if design.status == DesignStatus.FAILED:
            reason = design.error.message if design.error else "unknown error"
            raise RMFGError(200, "design_failed", f"Analysis failed: {reason}")
        return design

    # --- DFM ---------------------------------------------------------------

    async def create_dfm_report(
        self,
        design_id: str,
        configuration: ManufacturingConfiguration,
        generate_production_files: bool = True,
        idempotency_key: str = "",
    ) -> DFMReport:
        body = {
            "design_id": design_id,
            "configuration": configuration.to_payload(),
            "generate_production_files": generate_production_files,
        }
        data = await self._request(
            "POST", "/dfm", body=body, idempotency_key=idempotency_key
        )
        return DFMReport.model_validate(data)

    async def get_dfm_report(self, dfm_id: str) -> DFMReport:
        return DFMReport.model_validate(await self._request("GET", f"/dfm/{dfm_id}"))

    # --- quotes ------------------------------------------------------------

    async def create_quote(
        self,
        items: list[QuoteItemRequest],
        ship_to: Optional[ShipTo] = None,
        idempotency_key: str = "",
    ) -> Quote:
        data = await self._request(
            "POST",
            "/quotes",
            body=_basket(items, ship_to),
            idempotency_key=idempotency_key,
            wait=True,
        )
        return Quote.model_validate(data)

    async def get_quote(self, quote_id: str) -> Quote:
        return Quote.model_validate(await self._request("GET", f"/quotes/{quote_id}"))

    async def wait_for_quote(self, quote: Quote, timeout_seconds: float) -> Quote:
        """Poll until pricing and DFM have finished."""
        return await poll(
            lambda: self.get_quote(quote.id),
            initial=quote,
            is_pending=lambda q: q.status == QuoteStatus.PROCESSING,
            timeout_seconds=timeout_seconds,
            what=f"quote {quote.id}",
        )

    # --- carts -------------------------------------------------------------

    async def create_cart(
        self,
        items: list[QuoteItemRequest],
        ship_to: Optional[ShipTo] = None,
        shipping_option_id: str = "",
        client_reference_id: str = "",
        idempotency_key: str = "",
    ) -> Cart:
        body = _basket(items, ship_to)
        if shipping_option_id:
            body["shipping_option_id"] = shipping_option_id
        if client_reference_id:
            body["client_reference_id"] = client_reference_id
        data = await self._request(
            "POST", "/carts", body=body, idempotency_key=idempotency_key
        )
        return Cart.model_validate(data)

    async def get_cart(self, cart_id: str) -> Cart:
        return Cart.model_validate(await self._request("GET", f"/carts/{cart_id}"))

    async def update_cart(
        self,
        cart_id: str,
        *,
        items: Optional[list[QuoteItemRequest]] = None,
        ship_to: Optional[ShipTo] = None,
        shipping_option_id: str = "",
        idempotency_key: str = "",
    ) -> Cart:
        """Patch an open cart; omitted fields keep their current value."""
        body: dict[str, Any] = {}
        if items:
            body["items"] = [item.to_payload() for item in items]
        if ship_to:
            body["ship_to"] = ship_to.to_payload()
        if shipping_option_id:
            body["shipping_option_id"] = shipping_option_id
        data = await self._request(
            "PATCH", f"/carts/{cart_id}", body=body, idempotency_key=idempotency_key
        )
        return Cart.model_validate(data)

    async def pay_cart(
        self,
        cart_id: str,
        payment_type: PaymentType = PaymentType.CARD_ON_FILE,
        payment_method_id: str = "",
        customer_email: str = "",
        customer_phone: str = "",
        idempotency_key: str = "",
    ) -> Cart:
        payment: dict[str, str] = {"type": payment_type.value}
        if payment_type == PaymentType.PAYMENT_METHOD:
            payment["id"] = payment_method_id
        body: dict[str, Any] = {"payment": payment}
        if customer_email:
            body["customer_email"] = customer_email
        if customer_phone:
            body["customer_phone"] = customer_phone
        data = await self._request(
            "POST", f"/carts/{cart_id}/pay", body=body, idempotency_key=idempotency_key
        )
        return Cart.model_validate(data)

    # --- orders ------------------------------------------------------------

    async def get_order(self, order_id: str) -> Order:
        return Order.model_validate(await self._request("GET", f"/orders/{order_id}"))

    async def list_orders(
        self, limit: int = 20, cursor: str = ""
    ) -> tuple[list[Order], Optional[str]]:
        page = await self._request(
            "GET", "/orders", params={"limit": limit, "cursor": cursor}
        )
        orders = [Order.model_validate(row) for row in page.get("data") or []]
        return orders, page.get("next_cursor") if page.get("has_more") else None

    # --- review links ------------------------------------------------------

    async def create_review_link(
        self,
        design_id: str,
        configuration: ManufacturingConfiguration,
        dfm_id: str = "",
        idempotency_key: str = "",
    ) -> ReviewLink:
        body: dict[str, Any] = {
            "design_id": design_id,
            "configuration": configuration.to_payload(),
        }
        if dfm_id:
            body["dfm_id"] = dfm_id
        data = await self._request(
            "POST", "/review-links", body=body, idempotency_key=idempotency_key
        )
        return ReviewLink.model_validate(data)

    async def get_review_link(self, link_id: str) -> ReviewLink:
        data = await self._request("GET", f"/review-links/{link_id}")
        return ReviewLink.model_validate(data)


def _basket(items: list[QuoteItemRequest], ship_to: Optional[ShipTo]) -> dict:
    body: dict[str, Any] = {"items": [item.to_payload() for item in items]}
    if ship_to:
        body["ship_to"] = ship_to.to_payload()
    return body
