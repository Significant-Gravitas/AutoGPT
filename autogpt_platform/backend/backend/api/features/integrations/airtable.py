"""The Airtable bases a stored connection can reach, for the picker card that
replaces asking the user for an id from a URL."""

import logging
from typing import Annotated, Literal

import aiohttp
from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Depends, HTTPException, Path, Security, status
from pydantic import BaseModel

from backend.blocks.airtable._api import list_bases
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.model import Credentials, is_sdk_default
from backend.integrations.credentials_store import provider_matches
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.request import HTTPClientError, HTTPServerError

logger = logging.getLogger(__name__)

router = APIRouter()
creds_manager = IntegrationCredentialsManager()

# Airtable pages bases a thousand at a time; past this the picker says so.
_MAX_PAGES = 5


AirtableAccess = Literal["read", "edit", "create"]
# Airtable's base permission levels: a creator may change tables and fields,
# an editor may write records, and anyone else may only read.
_ACCESS: dict[str, AirtableAccess] = {"create": "create", "edit": "edit"}
_RANK: dict[AirtableAccess, int] = {"create": 0, "edit": 1, "read": 2}


class AirtableBase(BaseModel):
    id: str
    name: str
    # The most the connection may do here; the picker locks a base below what
    # the step needs.
    access: AirtableAccess


class AirtableBasesResponse(BaseModel):
    bases: list[AirtableBase]
    # More bases exist than one listing returns.
    truncated: bool = False


@router.get(
    "/credentials/{cred_id}/bases",
    summary="List the Airtable bases a connection can reach",
    operation_id="getV1ListAirtableBases",
    # It calls Airtable, so an account without a plan is stopped here too.
    dependencies=[Depends(enforce_payment_paywall)],
)
async def list_airtable_bases(
    cred_id: Annotated[str, Path(title="The Airtable credential to list with")],
    user_id: Annotated[str, Security(get_user_id)],
) -> AirtableBasesResponse:
    try:
        # Getting an OAuth credential may refresh it, which Airtable can refuse.
        credential = await _airtable_credential(user_id, cred_id)
    except HTTPClientError as e:
        logger.warning(f"Airtable refused to refresh this connection ({e.status_code})")
        raise _reconnect()
    except _UNREACHABLE as e:
        raise _down(e)
    try:
        bases, truncated = await _all_bases(credential)
    except HTTPClientError as e:
        logger.warning(f"Airtable refused to list bases ({e.status_code})")
        raise _listing_error(e.status_code)
    except _UNREACHABLE as e:
        raise _down(e)
    return AirtableBasesResponse(
        bases=sorted(bases, key=lambda b: (_RANK[b.access], b.name.casefold())),
        truncated=truncated,
    )


# Airtable failing or out of reach: not the user's to fix by reconnecting.
_UNREACHABLE = (HTTPServerError, aiohttp.ClientError, TimeoutError)


def _down(error: Exception) -> HTTPException:
    logger.warning(f"Airtable is unreachable: {type(error).__name__}")
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Airtable isn't answering right now. Try again in a minute.",
    )


def _reconnect() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "Airtable didn't accept this connection. Reconnect Airtable and try "
            "again."
        ),
    )


def _listing_error(airtable_status: int) -> HTTPException:
    """Only a refused connection is fixed by reconnecting; say what helps."""
    if airtable_status in (401, 403):
        return _reconnect()
    if airtable_status == 429:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Airtable is busy right now. Try again in a minute.",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Couldn't load your Airtable bases. Try again.",
    )


async def _airtable_credential(user_id: str, cred_id: str) -> Credentials:
    credential = (
        None if is_sdk_default(cred_id) else await creds_manager.get(user_id, cred_id)
    )
    if credential is None or not provider_matches(credential.provider, "airtable"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found"
        )
    return credential


async def _all_bases(credential: Credentials) -> tuple[list[AirtableBase], bool]:
    bases: list[AirtableBase] = []
    offset: str | None = None
    for _ in range(_MAX_PAGES):
        page = await list_bases(credential, offset=offset)
        bases += [
            AirtableBase(
                id=str(base["id"]),
                name=str(base.get("name") or "Untitled base"),
                access=_ACCESS.get(str(base.get("permissionLevel")), "read"),
            )
            for base in page.get("bases", [])
            if base.get("id")
        ]
        offset = page.get("offset")
        if not offset:
            return bases, False
    return bases, True
