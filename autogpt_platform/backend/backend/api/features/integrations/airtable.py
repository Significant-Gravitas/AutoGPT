"""The Airtable bases a stored connection can reach, for the picker card that
replaces asking the user for an id from a URL."""

import logging
from typing import Annotated

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, HTTPException, Path, Security, status
from pydantic import BaseModel

from backend.blocks.airtable._api import list_bases
from backend.data.model import Credentials, is_sdk_default
from backend.integrations.credentials_store import provider_matches
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.request import HTTPClientError

logger = logging.getLogger(__name__)

router = APIRouter()
creds_manager = IntegrationCredentialsManager()

# Airtable pages bases a thousand at a time; past this the picker says so.
_MAX_PAGES = 5
_EDITABLE = frozenset({"edit", "create"})


class AirtableBase(BaseModel):
    id: str
    name: str
    # Read-only bases are listed but cannot be picked for a workflow that writes.
    can_edit: bool


class AirtableBasesResponse(BaseModel):
    bases: list[AirtableBase]
    # More bases exist than one listing returns.
    truncated: bool = False


@router.get(
    "/credentials/{cred_id}/bases",
    summary="List the Airtable bases a connection can reach",
    operation_id="getV1ListAirtableBases",
)
async def list_airtable_bases(
    cred_id: Annotated[str, Path(title="The Airtable credential to list with")],
    user_id: Annotated[str, Security(get_user_id)],
) -> AirtableBasesResponse:
    credential = (
        None if is_sdk_default(cred_id) else await creds_manager.get(user_id, cred_id)
    )
    if credential is None or not provider_matches(credential.provider, "airtable"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found"
        )
    try:
        bases, truncated = await _all_bases(credential)
    except HTTPClientError as e:
        logger.warning(f"Airtable refused to list bases ({e.status_code})")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Airtable didn't accept this connection. Reconnect Airtable "
                "and try again."
            ),
        )
    return AirtableBasesResponse(
        bases=sorted(bases, key=lambda b: (not b.can_edit, b.name.casefold())),
        truncated=truncated,
    )


async def _all_bases(credential: Credentials) -> tuple[list[AirtableBase], bool]:
    bases: list[AirtableBase] = []
    offset: str | None = None
    for _ in range(_MAX_PAGES):
        page = await list_bases(credential, offset=offset)
        bases += [
            AirtableBase(
                id=str(base["id"]),
                name=str(base.get("name") or "Untitled base"),
                can_edit=base.get("permissionLevel") in _EDITABLE,
            )
            for base in page.get("bases", [])
            if base.get("id")
        ]
        offset = page.get("offset")
        if not offset:
            return bases, False
    return bases, True
