"""Client for the Printify REST API (https://developers.printify.com)."""

from typing import Any

from pydantic import ValidationError

from backend.blocks.printify._config import PrintifyCredentials
from backend.sdk import BaseModel, Field, Requests
from backend.util.request import HTTPClientError, HTTPServerError

API_URL = "https://api.printify.com/v1"
TOKEN_PAGE = "https://printify.com/app/account/api"
# Printify refuses requests without a User-Agent naming the client.
USER_AGENT = "AutoGPT"
# Throttling and 5xx are retried; without a cap a sustained one never ends.
MAX_ATTEMPTS = 3


class PrintifyError(HTTPClientError):
    """A 4xx from Printify, worded for the user. Keeps the shared HTTP error
    contract (and its status) so a rejected token still gets the reconnect
    card."""


class PrintifyShop(BaseModel):
    id: int
    title: str
    sales_channel: str = ""


class PrintifyBlueprint(BaseModel):
    id: int
    title: str
    brand: str = ""
    model: str = ""


class PrintifyPrintProvider(BaseModel):
    id: int
    title: str


class PrintifyVariant(BaseModel):
    id: int
    title: str
    options: dict[str, Any] = Field(default_factory=dict)


class PrintifyImage(BaseModel):
    id: str
    file_name: str = ""
    preview_url: str = ""
    width: int = 0
    height: int = 0


class PrintifyProduct(BaseModel):
    id: str
    title: str
    visible: bool = False
    is_locked: bool = False


class PrintifyClient:
    def __init__(self, credentials: PrintifyCredentials):
        self._token = credentials.api_key.get_secret_value()
        self._requests = Requests(
            raise_for_status=False, retry_max_attempts=MAX_ATTEMPTS
        )
        # Writes go out once: a 5xx after Printify already created the
        # product (or upload) would otherwise create it again.
        self._once = Requests(raise_for_status=False, retry_max_attempts=1)

    async def list_shops(self) -> list[PrintifyShop]:
        data = await self._request("GET", "/shops.json")
        return [PrintifyShop.model_validate(shop) for shop in data]

    async def upload_image(
        self, file_name: str, *, url: str | None = None, contents: str | None = None
    ) -> PrintifyImage:
        """Add an image to the merchant's media library, by URL or base64."""
        body = {"file_name": file_name, "url": url} if url else None
        if body is None:
            body = {"file_name": file_name, "contents": contents}
        data = await self._request("POST", "/uploads/images.json", json=body)
        return PrintifyImage.model_validate(data)

    async def list_blueprints(self) -> list[PrintifyBlueprint]:
        data = await self._request("GET", "/catalog/blueprints.json")
        return [PrintifyBlueprint.model_validate(item) for item in data]

    async def list_print_providers(
        self, blueprint_id: int
    ) -> list[PrintifyPrintProvider]:
        data = await self._request(
            "GET", f"/catalog/blueprints/{blueprint_id}/print_providers.json"
        )
        return [PrintifyPrintProvider.model_validate(item) for item in data]

    async def list_variants(
        self, blueprint_id: int, print_provider_id: int
    ) -> list[PrintifyVariant]:
        data = await self._request(
            "GET",
            f"/catalog/blueprints/{blueprint_id}/print_providers/"
            f"{print_provider_id}/variants.json",
        )
        return _VariantPage.model_validate(data).variants

    async def create_product(
        self, shop_id: int, product: dict[str, Any]
    ) -> PrintifyProduct:
        data = await self._request(
            "POST", f"/shops/{shop_id}/products.json", json=product
        )
        return PrintifyProduct.model_validate(data)

    async def publish_product(self, shop_id: int, product_id: str) -> None:
        """Push every part of the product to the shop's sales channel."""
        await self._request(
            "POST",
            f"/shops/{shop_id}/products/{product_id}/publish.json",
            json={
                "title": True,
                "description": True,
                "images": True,
                "variants": True,
                "tags": True,
                "keyFeatures": True,
                "shipping_template": True,
            },
        )

    async def list_products(
        self, shop_id: int, limit: int, page: int = 1
    ) -> list[PrintifyProduct]:
        data = await self._request(
            "GET",
            f"/shops/{shop_id}/products.json",
            params={"limit": limit, "page": page},
        )
        return _ProductPage.model_validate(data).data

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        transport = self._requests if method == "GET" else self._once
        response = await transport.request(
            method,
            f"{API_URL}{path}",
            headers={
                "Authorization": f"Bearer {self._token}",
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
            },
            json=json,
            params=params,
        )
        if response.ok:
            return response.json() if response.content else None
        message = _error_message(response.status, response.content)
        if response.status >= 500:
            raise HTTPServerError(message, response.status)
        raise PrintifyError(message, response.status)


def _error_message(status: int, body: bytes) -> str:
    if status == 401:
        return (
            "Printify did not accept the token. Create a new one at "
            f"{TOKEN_PAGE} and connect Printify again."
        )
    if status == 403:
        return (
            "This Printify token is missing a permission this step needs. "
            f"Create a token with full access at {TOKEN_PAGE}."
        )
    if status == 429:
        return "Printify is rate limiting requests. Try again in a minute."
    detail = _detail(body)
    return (
        f"Printify error {status}: {detail}" if detail else f"Printify error {status}"
    )


def _detail(body: bytes) -> str:
    """The most specific reason Printify gave, from its ``errors`` object."""
    try:
        error = _ErrorBody.model_validate_json(body)
    except ValidationError:
        return body.decode(errors="replace")[:300]
    reason = error.errors.reason if error.errors else ""
    return " ".join(part for part in (error.message, reason) if part)[:300]


class _ErrorReason(BaseModel):
    reason: str = ""


class _ErrorBody(BaseModel):
    message: str = ""
    errors: _ErrorReason | None = None


# Envelopes Printify wraps lists in; parsing them makes a malformed reply a
# validation error rather than a KeyError.
class _VariantPage(BaseModel):
    variants: list[PrintifyVariant]


class _ProductPage(BaseModel):
    data: list[PrintifyProduct]
