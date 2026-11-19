import asyncio
from functools import lru_cache
from typing import Literal
from uuid import UUID

import jwt
from pydantic import BaseModel, ValidationError

from backend.data.model import OAuth2Credentials

PERSONAL_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad"
MICROSOFT_PROVIDERS = frozenset({"microsoft"})


class MicrosoftIdentity(BaseModel):
    tenant_id: str
    account_id: str
    account_type: Literal["personal", "work"]


async def populate_microsoft_identity(
    credential: OAuth2Credentials, id_token: str, client_id: str
) -> None:
    claims = await asyncio.to_thread(_validate_identity_token, id_token, client_id)
    tenant = str(UUID(claims["tid"]))
    identity = MicrosoftIdentity(
        tenant_id=tenant,
        account_id=claims.get("oid") or claims["sub"],
        account_type="personal" if tenant == PERSONAL_TENANT_ID else "work",
    )
    credential.metadata["microsoft_identity"] = identity.model_dump()
    credential.metadata["microsoft_account_id"] = identity.account_id
    credential.username = claims.get("preferred_username") or credential.username
    name = claims.get("name") or credential.username or "Microsoft"
    credential.title = f"{name} ({identity.account_type} account)"


def microsoft_identity(credential: OAuth2Credentials) -> MicrosoftIdentity | None:
    data = credential.metadata.get("microsoft_identity")
    if not data:
        return None
    try:
        return MicrosoftIdentity.model_validate(data)
    except ValidationError:
        return None


def same_microsoft_account(
    existing: OAuth2Credentials, incoming: OAuth2Credentials
) -> bool:
    left, right = microsoft_identity(existing), microsoft_identity(incoming)
    if left and right:
        return (left.tenant_id, left.account_id) == (right.tenant_id, right.account_id)
    left_id = existing.metadata.get("microsoft_account_id")
    right_id = incoming.metadata.get("microsoft_account_id")
    return bool(left_id and right_id and left_id == right_id)


@lru_cache(maxsize=32)
def _signing_keys(tenant: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(
        f"https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys",
        timeout=15,
    )


def _validate_identity_token(id_token: str, client_id: str) -> dict:
    unverified = jwt.decode(id_token, options={"verify_signature": False})
    tenant = str(UUID(unverified["tid"]))
    key = _signing_keys(tenant).get_signing_key_from_jwt(id_token).key
    return jwt.decode(
        id_token,
        key,
        algorithms=["RS256"],
        audience=client_id,
        issuer=f"https://login.microsoftonline.com/{tenant}/v2.0",
        options={"require": ["aud", "iss", "exp", "sub", "tid"]},
    )
