"""HTTP access to the ChatGPT Codex backend.

Everything the Codex CLI used to broker -- the model catalog, rate limits,
account identity and inference itself -- is reachable over ordinary HTTPS, so
no binary, no subprocess and no synthetic ``$HOME`` are involved.

Inference goes through the stock OpenAI SDK: the endpoint *is* the Responses
API, so ``AsyncOpenAI`` pointed at the ChatGPT base URL streams, calls tools and
reports usage without any bespoke SSE handling.
"""

import logging
import uuid
from typing import Any, Mapping

from openai import AsyncOpenAI

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import decode_jwt_claims
from backend.integrations.codex.chatgpt_auth import ORIGINATOR, USER_AGENT
from backend.integrations.codex.credential_codec import bundle_from_credentials
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexModelInfo,
    CodexRateLimitsSnapshot,
    CodexRateLimitWindow,
    CodexReasoningEffort,
)
from backend.util.request import Requests

logger = logging.getLogger(__name__)

API_BASE = "https://chatgpt.com/backend-api/codex"
MODELS_URL = f"{API_BASE}/models"

# ``/models`` requires the parameter but does not validate it: the catalog is
# identical for "0.0.0" and a current CLI build, so this stays a constant rather
# than a version we would have to chase.
CLIENT_VERSION = "0.0.0"

_VALID_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)


def build_client(credentials: OAuth2Credentials) -> AsyncOpenAI:
    """Build a client bound to one user's subscription.

    Deliberately constructed per call and never memoised. An ``AsyncOpenAI``
    instance carries the credential it was built with, so a cached client would
    silently bill every subsequent turn in the same worker to whichever user
    happened to create it first.
    """
    return AsyncOpenAI(
        api_key=credentials.access_token.get_secret_value(),
        base_url=API_BASE,
        default_headers=_inference_headers(credentials),
    )


def _inference_headers(credentials: OAuth2Credentials) -> dict[str, str]:
    headers = {
        "originator": ORIGINATOR,
        "user-agent": USER_AGENT,
        "session_id": str(uuid.uuid4()),
    }
    account_id = account_id_for(credentials)
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def account_id_for(credentials: OAuth2Credentials) -> str | None:
    try:
        bundle = bundle_from_credentials(credentials)
        return decode_jwt_claims(bundle.tokens.id_token).chatgpt_account_id
    except Exception:
        logger.warning("Could not read the account id from a ChatGPT token")
        return None


def account_snapshot(credentials: OAuth2Credentials) -> CodexAccountSnapshot:
    """Derive account identity from the token itself; there is no account API."""
    try:
        bundle = bundle_from_credentials(credentials)
        claims = decode_jwt_claims(bundle.tokens.id_token)
    except Exception:
        return CodexAccountSnapshot(connected=False, requires_openai_auth=True)
    return CodexAccountSnapshot(
        connected=True,
        requires_openai_auth=False,
        account_type="chatgpt",
        email=claims.email,
        plan_type=claims.plan_type,
    )


async def fetch_models(credentials: OAuth2Credentials) -> list[CodexModelInfo]:
    response = await Requests().get(
        f"{MODELS_URL}?client_version={CLIENT_VERSION}",
        headers={
            "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
            "accept": "application/json",
            **_inference_headers(credentials),
        },
    )
    payload = response.json()
    entries = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []

    models = [parsed for entry in entries if (parsed := _parse_model(entry))]
    # ``priority`` orders the picker; the lowest listed one is what Codex
    # itself preselects. Nothing in the payload flags a default directly.
    default_slug = next(
        (
            model.model
            for model in sorted(
                (m for m in models if not m.hidden),
                key=lambda m: _priority_of(entries, m.model),
            )
        ),
        None,
    )
    return [
        model.model_copy(update={"is_default": model.model == default_slug})
        for model in models
    ]


def _priority_of(entries: list[Any], slug: str) -> int:
    for entry in entries:
        if isinstance(entry, dict) and entry.get("slug") == slug:
            priority = entry.get("priority")
            if isinstance(priority, int):
                return priority
    return 1_000


def _parse_model(entry: Any) -> CodexModelInfo | None:
    if not isinstance(entry, dict):
        return None
    slug = entry.get("slug")
    if not isinstance(slug, str) or not slug:
        return None

    efforts = _supported_efforts(entry.get("supported_reasoning_levels"))
    default_effort = entry.get("default_reasoning_level")
    if default_effort not in _VALID_EFFORTS:
        default_effort = efforts[0] if efforts else "medium"

    modalities = entry.get("input_modalities")
    return CodexModelInfo(
        model=slug,
        display_name=str(entry.get("display_name") or slug),
        # Overwritten by the caller once the full list has been ranked.
        is_default=False,
        hidden=entry.get("visibility") == "hide",
        default_reasoning_effort=default_effort,  # type: ignore[arg-type]
        supported_reasoning_efforts=efforts,
        input_modalities=(
            [str(m) for m in modalities] if isinstance(modalities, list) else []
        ),
    )


def _supported_efforts(raw: Any) -> list[CodexReasoningEffort]:
    """Levels arrive as ``[{"effort": "low", "description": ...}, ...]``."""
    if not isinstance(raw, list):
        return []
    efforts: list[CodexReasoningEffort] = []
    for level in raw:
        effort = level.get("effort") if isinstance(level, dict) else level
        if isinstance(effort, str) and effort in _VALID_EFFORTS:
            efforts.append(effort)  # type: ignore[arg-type]
    return efforts


# --------------------------------------------------------------------------- #
# Rate limits
#
# The backend reports quota on the response headers of every inference call, so
# there is nothing extra to poll -- the numbers arrive with the work that spent
# them, and are always current.
# --------------------------------------------------------------------------- #


def parse_rate_limits(headers: Mapping[str, str]) -> CodexRateLimitsSnapshot:
    lookup = {key.lower(): value for key, value in headers.items()}
    return CodexRateLimitsSnapshot(
        plan_type=lookup.get("x-codex-plan-type"),
        limit_id=lookup.get("x-codex-active-limit"),
        limit_name=lookup.get("x-codex-limit-name"),
        primary=_window(lookup, "primary"),
        secondary=_window(lookup, "secondary"),
        has_credits=_bool(lookup.get("x-codex-credits-has-credits")),
        unlimited_credits=_bool(lookup.get("x-codex-credits-unlimited")),
    )


def _window(lookup: Mapping[str, str], scope: str) -> CodexRateLimitWindow | None:
    used = _int(lookup.get(f"x-codex-{scope}-used-percent"))
    window = _int(lookup.get(f"x-codex-{scope}-window-minutes"))
    # A zero-length window is how an inactive tier is reported, not a real one.
    if used is None or not window:
        return None
    return CodexRateLimitWindow(
        used_percent=used,
        window_duration_mins=window,
        resets_at=_int(lookup.get(f"x-codex-{scope}-reset-at")),
    )


def _int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _bool(value: str | None) -> bool | None:
    """Sent title-cased ("False"), so a plain ``== "true"`` would misread it."""
    if value is None or value == "":
        return None
    return value.strip().lower() == "true"
