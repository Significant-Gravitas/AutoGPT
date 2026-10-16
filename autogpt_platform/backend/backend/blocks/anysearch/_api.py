from enum import Enum
from typing import Any
from urllib.parse import quote

from backend.sdk import APIKeyCredentials, BaseModel, Requests, SchemaField

from ._http import raise_for_status

ANYSEARCH_API_URL = "https://api.anysearch.com"

# Sent on every request, anonymous or keyed, for AnySearch's version-based
# traffic attribution (their server-side analytics; upstream issue #40).
# Static constant on purpose - it is not tied to the backend package version.
ANYSEARCH_CLIENT = "autogpt-platform/1.0.0"


class AnySearchDomain(Enum):
    """Vertical domains accepted by the AnySearch domain search parameter."""

    ACADEMIC = "academic"
    AGRICULTURE = "agriculture"
    BUSINESS = "business"
    CODE = "code"
    ENERGY = "energy"
    ENVIRONMENT = "environment"
    FILM = "film"
    FINANCE = "finance"
    GAMING = "gaming"
    GENERAL = "general"
    HEALTH = "health"
    IP = "ip"
    LEGAL = "legal"
    RESOURCE = "resource"
    SECURITY = "security"
    SOCIAL_MEDIA = "social_media"
    TRAVEL = "travel"


class AnySearchAuth(str, Enum):
    """How an AnySearch block authenticates.

    API_KEY uses a stored AnySearch credential; ANONYMOUS sends
    unauthenticated requests at the vendor's lower rate limit and makes the
    credentials field optional - the same discriminator mechanism the
    AutoPilot block uses for its credential-free transport.
    """

    API_KEY = "api_key"
    ANONYMOUS = "anonymous"

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        json_schema = handler(schema)
        json_schema["enumNames"] = ["API key", "Anonymous (rate-limited)"]
        return json_schema


class AnySearchResult(BaseModel):
    """Schema for a single AnySearch search result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(
        description="Query-relevant snippet of the result page", default=""
    )
    content: str | None = SchemaField(
        description="Page content returned for the result, when available",
        default=None,
    )


class AnySearchQueryResults(BaseModel):
    """Results produced for one query of a parallel AnySearch batch."""

    query: str = SchemaField(description="The query this result group belongs to")
    results: list[AnySearchResult] = SchemaField(
        description="Results for this query", default_factory=list
    )
    error: str = SchemaField(
        description="Error message if this query failed", default=""
    )
    request_id: str = SchemaField(
        description="Server-side request ID for this query", default=""
    )


class AnySearchEnvelope(BaseModel):
    """Decoded AnySearch ``{code, message, data, request_id}`` envelope."""

    data: dict[str, Any]
    request_id: str = ""


class AnySearchClient:
    """Thin async REST client for api.anysearch.com.

    Covers POST /v1/search + /v1/extract and the public GET catalog
    endpoints /v1/domains + /v1/sub-domains. The service answers a
    {code, message, data, request_id} envelope; a non-zero code is an
    API-level error even on HTTP 200. No public per-call pricing is
    published, so callers record no provider_cost.
    """

    def __init__(self, credentials: APIKeyCredentials | None):
        # No credential selected (anonymous tier) or an empty key value:
        # omit the Authorization header rather than sending an empty Bearer.
        # X-Anysearch-Client goes out on both modes for traffic attribution.
        api_key = credentials.api_key.get_secret_value().strip() if credentials else ""
        headers = {"X-Anysearch-Client": ANYSEARCH_CLIENT}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.requests = Requests(
            trusted_origins=[ANYSEARCH_API_URL],
            extra_headers=headers,
            raise_for_status=False,
            retry_max_attempts=1,
        )

    async def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.requests.post(
            f"{ANYSEARCH_API_URL}/v1/search", json=payload
        )
        raise_for_status(response)
        if not response.ok:
            raise ValueError(
                f"AnySearch search request failed with HTTP {response.status}"
            )
        return response.json()

    async def extract(self, url: str) -> dict[str, Any]:
        response = await self.requests.post(
            f"{ANYSEARCH_API_URL}/v1/extract", json={"url": url}
        )
        raise_for_status(response)
        if not response.ok:
            raise ValueError(
                f"AnySearch extract request failed with HTTP {response.status}"
            )
        return response.json()

    async def domains(self) -> dict[str, Any]:
        """GET /v1/domains - public catalog; free of auth and quota cost."""
        response = await self.requests.get(f"{ANYSEARCH_API_URL}/v1/domains")
        raise_for_status(response)
        if not response.ok:
            raise ValueError(
                f"AnySearch domains request failed with HTTP {response.status}"
            )
        return response.json()

    async def sub_domains(self, domains: list[str]) -> dict[str, Any]:
        """GET /v1/sub-domains - sub-domain catalog with parameter contracts.

        ``domain`` repeats per value (``?domain=a&domain=b``); the endpoint
        rejects comma-joined lists and requires at least one domain.
        """
        if not domains:
            raise ValueError("sub_domains requires at least one domain")
        query = "&".join(f"domain={quote(d, safe='')}" for d in domains)
        response = await self.requests.get(
            f"{ANYSEARCH_API_URL}/v1/sub-domains?{query}"
        )
        raise_for_status(response)
        if not response.ok:
            raise ValueError(
                "AnySearch sub-domains request failed " f"with HTTP {response.status}"
            )
        return response.json()


# Substrings that mark a message as carrying credential material; on a match
# the upstream text is replaced by a generic one (a 402 anonymous-quota reply
# can echo back plaintext username/password/api_key - official warning; an
# echoed Authorization/Bearer header is credential material too).
_SENSITIVE_MARKERS = (
    "password",
    "api_key",
    "apikey",
    "secret",
    "token",
    "authorization",
    "bearer",
)


def sanitize_upstream_text(text: str) -> str:
    """Return ``text`` verbatim unless it appears to carry credential
    material - envelope messages and HTTP-error bodies alike."""
    if any(m in text.lower() for m in _SENSITIVE_MARKERS):
        return "upstream returned an error (details withheld)"
    return text


def unwrap_envelope(
    response: dict[str, Any], required: str | None = None
) -> AnySearchEnvelope:
    """Decode the AnySearch envelope; raise on API-level errors.

    ``required`` names a ``data`` key the caller's contract needs - a
    successful response missing it is a response error, not an empty result.
    Raised errors always carry the server-side request_id.
    """
    if not isinstance(response, dict):
        raise ValueError("malformed AnySearch response envelope")
    raw_rid = response.get("request_id")
    if raw_rid is not None and not isinstance(raw_rid, str):
        raise ValueError("malformed AnySearch response envelope (request_id)")
    rid = raw_rid or ""
    raw_message = response.get("message")
    if raw_message is not None and not isinstance(raw_message, str):
        raise ValueError(f"malformed AnySearch response envelope (request_id: {rid})")
    code = response.get("code")
    data = response.get("data")
    if isinstance(code, bool) or not isinstance(code, int):
        # False == 0 in Python; the contract uses numeric codes only.
        raise ValueError(f"malformed AnySearch response envelope (request_id: {rid})")
    if code != 0:
        message = sanitize_upstream_text(raw_message or "unknown error")
        raise ValueError(f"AnySearch API error: {message} (request_id: {rid})")
    if not isinstance(data, dict):
        # code==0 with a non-dict body is a malformed success, not an API
        # error - do not surface the upstream "success" message as one.
        raise ValueError(f"malformed AnySearch response envelope (request_id: {rid})")
    if required is not None and required not in data:
        raise ValueError(
            f"AnySearch response missing data.{required} (request_id: {rid})"
        )
    return AnySearchEnvelope(data=data, request_id=rid)


def result_from_dict(r: dict[str, Any]) -> AnySearchResult:
    if not isinstance(r, dict):
        raise ValueError("malformed result row in AnySearch response")
    return AnySearchResult(
        title=r.get("title") or "",
        url=r.get("url") or "",
        snippet=r.get("snippet") or "",
        content=r.get("content"),
    )


def check_vertical_inputs(
    domain: AnySearchDomain | None, sub_domain: str | None
) -> None:
    """Shared domain/sub_domain consistency check for the search blocks.

    domain is validated locally only - it is never sent to the API.
    """
    if domain and not sub_domain:
        raise ValueError("sub_domain is required when domain is set")
    if domain and sub_domain and not sub_domain.startswith(f"{domain.value}."):
        raise ValueError(
            "sub_domain must belong to the selected domain " f"({domain.value}.*)"
        )
