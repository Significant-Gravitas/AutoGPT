"""The shared key that authenticates calls between internal services.

A method an ``AppService`` exposes runs for anything that can reach the
service's port, and many of them read or change any user's data. So every
exposed method requires ``INTERNAL_API_KEY``, which ``get_service_client``
sends with every call. It travels as a bearer token in ``Authorization``,
whose value Sentry and the event scrubber in ``backend.util.metrics`` already
keep out of error reports. Health checks and ``/metrics`` stay open: probes
and the Prometheus scraper call them, and neither returns user data.

When the key is enforced:

- ``BEHAVE_AS=local`` (self-hosted and development): whenever it is set.
- ``BEHAVE_AS=cloud``: only once ``INTERNAL_RPC_REQUIRE`` is on. A hosted
  deploy replaces services gradually, and an instance of the previous release
  keeps calling without the key for as long as it drains (an executor for up
  to a day), so the key is sent for one release before it is required.

With no key set at all, calls are accepted as before and every service warns.
"""

import hmac
import logging
from typing import Annotated

from fastapi import HTTPException, Request, Security, params
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.util.exceptions import MissingConfigError
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False, scheme_name="InternalServiceKey")


def internal_auth_headers(settings: Settings | None = None) -> dict[str, str]:
    """The headers an RPC client sends with every call: the key, if set."""
    key = (settings or Settings()).secrets.internal_api_key
    return {"Authorization": f"Bearer {key}"} if key else {}


def internal_auth_dependencies(
    service_name: str, settings: Settings | None = None
) -> list[params.Depends]:
    """What every exposed method of *service_name* depends on: the key check,
    when it is enforced, and nothing otherwise.

    Raises ``MissingConfigError`` when ``INTERNAL_RPC_REQUIRE`` is on but no key
    is set, so the service refuses to start rather than serve unauthenticated.
    """
    settings = settings or Settings()
    key = settings.secrets.internal_api_key
    required = settings.config.internal_rpc_require
    if required and not key:
        raise MissingConfigError(
            "INTERNAL_RPC_REQUIRE is on but INTERNAL_API_KEY is not set: give "
            "every backend service of this deployment the same key."
        )
    if not key:
        logger.warning(
            f"[SECURITY] [{service_name}] INTERNAL_API_KEY is not set: anything "
            "that can reach this service's port can call its methods. Add a line "
            "`INTERNAL_API_KEY=` to autogpt_platform/backend/.env and run "
            "`make init-env` in autogpt_platform/ to generate one."
        )
        return []
    if settings.config.behave_as == BehaveAs.CLOUD and not required:
        logger.warning(
            f"[SECURITY] [{service_name}] INTERNAL_API_KEY is sent but not yet "
            "required. Turn on INTERNAL_RPC_REQUIRE once no instance started "
            "without the key is still running."
        )
        return []
    return [Security(_verify_key(key, service_name))]


def _verify_key(key: str, service_name: str):
    # Bytes, so a header holding non-ASCII text is refused instead of making
    # compare_digest raise.
    expected = key.encode()

    async def verify(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    ) -> None:
        supplied = credentials.credentials.encode() if credentials else b""
        if hmac.compare_digest(supplied, expected):
            return
        caller = request.client.host if request.client else "unknown"
        logger.warning(
            f"[{service_name}] Refused {request.url.path} from {caller}: "
            "missing or wrong internal service key"
        )
        raise HTTPException(status_code=401, detail="Invalid internal service key")

    return verify
