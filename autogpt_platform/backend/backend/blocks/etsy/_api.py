"""Client for the Etsy Open API v3 (https://developers.etsy.com)."""

from io import BytesIO
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.sdk import OAuth2Credentials, Requests
from backend.util.request import HTTPClientError, HTTPServerError, Response
from backend.util.settings import Secrets

API_URL = "https://api.etsy.com/v3/application"


# Throttling and 5xx are retried; without a cap a sustained one never ends.
MAX_ATTEMPTS = 3
# No shop on the account is the user's to fix, like a missing resource.
NO_SHOP_STATUS = 404


class EtsyError(HTTPClientError):
    """A 4xx from Etsy, worded for the user. Keeps the shared HTTP error
    contract (and its status) so a rejected token, or a refused refresh,
    still gets the reconnect card."""


class EtsyMoney(BaseModel):
    amount: int
    divisor: int
    currency_code: str


class EtsyShop(BaseModel):
    shop_id: int
    shop_name: str
    title: str | None = None
    url: str = ""
    currency_code: str = ""


class EtsyShippingProfile(BaseModel):
    shipping_profile_id: int
    title: str = ""


class EtsyProcessingProfile(BaseModel):
    """How long the shop takes to make or ship an item (Etsy's "readiness
    state" definitions)."""

    readiness_state_id: int
    readiness_state: str = ""
    min_processing_days: int | None = None
    max_processing_days: int | None = None
    label: str = Field(default="", alias="processing_days_display_label")

    @model_validator(mode="after")
    def _label_from_days(self) -> "EtsyProcessingProfile":
        """Etsy's schema marks no field required: without its translated
        label, say the range in days the way Etsy writes it ("1-3 days")."""
        if not self.label and self.min_processing_days is not None:
            low = self.min_processing_days
            high = self.max_processing_days or low
            unit = "day" if high == 1 else "days"
            self.label = f"{low} {unit}" if low == high else f"{low}-{high} {unit}"
        return self


class EtsyReturnPolicy(BaseModel):
    """A shop's return policy. Outside the EU, a physical listing needs one
    before it can be published."""

    return_policy_id: int
    accepts_returns: bool = False
    accepts_exchanges: bool = False
    return_deadline: int | None = None
    """Days the buyer has to return or exchange, when the shop takes them."""


class EtsyCategory(BaseModel):
    id: int
    name: str
    path: str


class EtsyListing(BaseModel):
    listing_id: int
    title: str
    state: str = ""
    url: str = ""
    quantity: int = 0
    tags: list[str] = Field(default_factory=list)
    price: EtsyMoney | None = None


class EtsyListingImage(BaseModel):
    listing_image_id: int
    url_fullxfull: str = ""


def api_key_header(client_id: str, client_secret: str) -> dict[str, str]:
    """Etsy wants the app's keystring and shared secret on every request."""
    return {"x-api-key": f"{client_id}:{client_secret}"}


def raise_for_etsy_error(response: Response) -> None:
    if response.ok:
        return
    message = _error_message(response.status, response.content)
    if response.status >= 500:
        raise HTTPServerError(message, response.status)
    raise EtsyError(message, response.status)


class EtsyClient:
    def __init__(
        self,
        credentials: OAuth2Credentials,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        secrets = Secrets()
        self._headers = {
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
            **api_key_header(
                client_id or secrets.etsy_client_id,
                client_secret or secrets.etsy_client_secret,
            ),
        }
        self._requests = self.raw_requests()
        # Creates go out once: a 5xx after Etsy already made the draft (or
        # took the photo) would otherwise make a second one.
        self._once = self.raw_requests(attempts=1)

    @staticmethod
    def raw_requests(attempts: int = MAX_ATTEMPTS) -> Requests:
        return Requests(raise_for_status=False, retry_max_attempts=attempts)

    async def get_my_shop(self) -> EtsyShop:
        me = _Me.model_validate(await self._call("GET", "/users/me"))
        if not me.shop_id:
            raise EtsyError(
                "This Etsy account has no shop yet. Open one on Etsy first.",
                NO_SHOP_STATUS,
            )
        return EtsyShop.model_validate(await self._call("GET", f"/shops/{me.shop_id}"))

    async def list_shipping_profiles(self, shop_id: int) -> list[EtsyShippingProfile]:
        data = await self._call("GET", f"/shops/{shop_id}/shipping-profiles")
        return _ShippingProfilePage.model_validate(data).results

    async def list_processing_profiles(
        self, shop_id: int
    ) -> list[EtsyProcessingProfile]:
        data = await self._call("GET", f"/shops/{shop_id}/readiness-state-definitions")
        return _ProcessingProfilePage.model_validate(data).results

    async def list_return_policies(self, shop_id: int) -> list[EtsyReturnPolicy]:
        data = await self._call("GET", f"/shops/{shop_id}/policies/return")
        return _ReturnPolicyPage.model_validate(data).results

    async def list_categories(self) -> list[EtsyCategory]:
        """Every seller category, flattened, each with its full path."""
        data = await self._call("GET", "/seller-taxonomy/nodes")
        return _flatten_categories(_TaxonomyPage.model_validate(data).results, [])

    async def create_draft_listing(
        self, shop_id: int, form: dict[str, str]
    ) -> EtsyListing:
        data = await self._call("POST", f"/shops/{shop_id}/listings", form=form)
        return EtsyListing.model_validate(data)

    async def upload_listing_image(
        self,
        shop_id: int,
        listing_id: int,
        image: tuple[str, bytes, str],
        form: dict[str, str],
    ) -> EtsyListingImage:
        filename, content, mime_type = image
        data = await self._call(
            "POST",
            f"/shops/{shop_id}/listings/{listing_id}/images",
            form=form,
            files=[("image", (filename, BytesIO(content), mime_type))],
        )
        return EtsyListingImage.model_validate(data)

    async def list_listings(
        self, shop_id: int, state: str, limit: int
    ) -> list[EtsyListing]:
        data = await self._call(
            "GET",
            f"/shops/{shop_id}/listings",
            params={"state": state, "limit": limit},
        )
        return _ListingPage.model_validate(data).results

    async def update_listing(
        self, shop_id: int, listing_id: int, form: dict[str, str]
    ) -> EtsyListing:
        data = await self._call(
            "PATCH", f"/shops/{shop_id}/listings/{listing_id}", form=form
        )
        return EtsyListing.model_validate(data)

    async def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        form: dict[str, str] | None = None,
        files: list[tuple[str, tuple[str, BytesIO, str]]] | None = None,
    ) -> Any:
        retryable = method in ("GET", "PATCH")
        transport = self._requests if retryable else self._once
        response = await transport.request(
            method,
            f"{API_URL}{path}",
            headers=self._headers,
            params=params,
            data=form,
            files=files,
        )
        raise_for_etsy_error(response)
        return response.json()


def csv(values: list[str]) -> str:
    """A list field as Etsy's form encoding expects it."""
    return ",".join(values)


def flag(value: bool) -> str:
    return "true" if value else "false"


def _flatten_categories(
    nodes: list["_TaxonomyNode"], parents: list[str]
) -> list[EtsyCategory]:
    categories: list[EtsyCategory] = []
    for node in nodes:
        path = [*parents, node.name]
        categories.append(
            EtsyCategory(id=node.id, name=node.name, path=" > ".join(path))
        )
        categories += _flatten_categories(node.children, path)
    return categories


def _error_message(status: int, body: bytes) -> str:
    if status == 401:
        return "Etsy signed this connection out. Connect Etsy again."
    if status == 403:
        return (
            "Etsy refused this request. Reconnect Etsy and allow access to "
            "your shop and listings."
        )
    if status == 429:
        return "Etsy is rate limiting requests. Try again in a minute."
    detail = _detail(body)
    return f"Etsy error {status}: {detail}" if detail else f"Etsy error {status}"


def _detail(body: bytes) -> str:
    try:
        return _ErrorBody.model_validate_json(body).error[:300]
    except ValidationError:
        return body.decode(errors="replace")[:300]


class _ErrorBody(BaseModel):
    error: str


class _Me(BaseModel):
    shop_id: int | None = None


class _TaxonomyNode(BaseModel):
    id: int
    name: str
    children: list["_TaxonomyNode"] = Field(default_factory=list)


# Etsy wraps every list in {"count", "results"}; parsing the envelope makes a
# malformed reply a validation error rather than a KeyError.
class _ShippingProfilePage(BaseModel):
    results: list[EtsyShippingProfile]


class _ProcessingProfilePage(BaseModel):
    results: list[EtsyProcessingProfile]


class _ReturnPolicyPage(BaseModel):
    results: list[EtsyReturnPolicy]


class _TaxonomyPage(BaseModel):
    results: list[_TaxonomyNode]


class _ListingPage(BaseModel):
    results: list[EtsyListing]
