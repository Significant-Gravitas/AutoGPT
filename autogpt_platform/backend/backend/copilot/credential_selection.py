"""The credential a user picked for each provider, for the rest of a chat.

The connect card and the backend used to choose credentials separately: the
card showed the account the user picked, and the backend re-matched on its own
and took whichever qualifying credential was stored first. With two accounts
for one provider the run could land on the wrong one.

So the card now reports the pick, and it is kept here for the session. A chat
tool uses exactly that credential, and when several credentials qualify and
none was picked it asks instead of choosing (see
``copilot.tools.utils.find_matching_credential``).

Kept in Redis rather than on the session row because it is advisory state with
a safe failure: losing it means the user is asked again, never that a run uses
an account they did not choose.

A schedule made from a chat keeps the picks too, as *pins* stored on the
schedule (SECRT-2804). Its turns run later with nobody to ask, often in a chat
that never saw the pick, so the executor hands the schedule's pins to each
turn it fires and they win over anything picked in the chat.
"""

import logging
from collections.abc import Mapping
from contextvars import ContextVar
from urllib.parse import urlsplit

from pydantic import BaseModel

from backend.data.model import Credentials
from backend.data.redis_client import get_redis_async
from backend.integrations.credentials_store import provider_matches

logger = logging.getLogger(__name__)

SELECTION_TTL = 30 * 86400  # 30 days, far longer than a chat stays active


class CredentialPin(BaseModel):
    """The account a schedule runs one provider on, chosen when it was made.

    The title is kept so a run can still name the account after it was
    deleted, which is when the name matters most.
    """

    id: str
    title: str = ""


CredentialPins = dict[str, CredentialPin]
"""``{provider: pin}`` stored on a schedule."""

_current_turn_pins: ContextVar[Mapping[str, CredentialPin]] = ContextVar(
    "_current_turn_pins", default={}
)


def set_turn_credential_pins(pins: Mapping[str, CredentialPin] | None) -> None:
    """Record the pins of the schedule that fired the running turn. The
    executor sets it for every turn, so pins never carry over to the next."""
    _current_turn_pins.set(dict(pins or {}))


def turn_credential_pins() -> Mapping[str, CredentialPin]:
    """``{provider: pin}`` for the running turn; empty unless a schedule with
    pins fired it."""
    return _current_turn_pins.get()


def _key(session_id: str) -> str:
    return f"copilot:credential-selection:{session_id}"


def host_selection_key(provider: str, url: str) -> str:
    try:
        parsed = urlsplit(url if "://" in url else f"http://{url}")
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return provider
    if not host:
        return provider
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if parsed.scheme == "https" else 80
    if port is not None and port != default_port:
        host = f"{host}:{port}"
    return f"{provider}:{host.lower()}"


def credential_matches_selection(credential: Credentials, key: str) -> bool:
    provider, separator, host = key.partition(":")
    if not provider_matches(credential.provider, provider):
        return False
    if not separator:
        return True
    if credential.type != "host_scoped" or not host:
        return False
    try:
        parsed = urlsplit(f"//{host}")
        if parsed.netloc != host or parsed.username is not None:
            return False
        port = parsed.port
    except ValueError:
        return False
    urls = (
        [f"http://{host}", f"https://{host}"]
        if port is not None
        else [f"http://{host}:80", f"https://{host}:443"]
    )
    return any(credential.matches_url(url) for url in urls)


async def remember_selection(session_id: str, selections: dict[str, str]) -> None:
    """Record ``{provider: credential_id}`` picks, replacing earlier picks for
    the same providers. Raises if Redis is unreachable: the caller is an API
    request and should report that the pick did not take."""
    if not selections:
        return
    redis = await get_redis_async()
    # One transaction: a failure part way through must not leave half a pick,
    # which later matching would apply as if it were the whole of it.
    async with redis.pipeline(transaction=True) as pipe:
        for provider, credential_id in selections.items():
            pipe.hset(_key(session_id), provider, credential_id)
        pipe.expire(_key(session_id), SELECTION_TTL)
        await pipe.execute()


async def selected_credentials(session_id: str | None) -> dict[str, str]:
    """``{provider: credential_id}`` picked in this session; empty when none.

    The pins of the schedule that fired the running turn are laid over the
    chat's picks: the schedule was set up to run on those accounts, whatever
    the chat it lands in picked since.

    Never raises. An unreadable selection degrades to "nothing picked", which
    makes the tools ask rather than guess.
    """
    pinned = {provider: pin.id for provider, pin in turn_credential_pins().items()}
    if not session_id:
        return pinned
    try:
        redis = await get_redis_async()
        raw = await redis.hgetall(_key(session_id))
    except Exception:
        logger.warning("Could not read credential selection for session %s", session_id)
        return pinned
    picked = {
        (k.decode() if isinstance(k, bytes) else k): (
            v.decode() if isinstance(v, bytes) else v
        )
        for k, v in raw.items()
    }
    return {**picked, **pinned}
