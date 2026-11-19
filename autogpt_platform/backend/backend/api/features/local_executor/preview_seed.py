"""The Local PC companion's OAuth app, registered in a preview's database.

`autogpt-shim auth` signs in through the public OAuth client named by
``ChatConfig.local_pc_executor_oauth_client_id``. OAuth apps can't be
registered through the API, so a real deployment registers this one by hand
(``oauth-tool generate-app`` with ``--public``). A preview starts from a
fixture without it, which left nothing to link a computer through, so the
preview seed (``seed-preview-marketplace``) registers it, owned by the seed's
admin, with the companion's loopback callbacks.
"""

import secrets
import uuid

from autogpt_libs.api_key.keysmith import APIKeySmith
from prisma.enums import APIKeyPermission
from prisma.models import OAuthApplication

from backend.copilot.config import ChatConfig

# `autogpt-shim auth` listens on the first free port of these.
CALLBACK_PORTS = range(41899, 41911)
REDIRECT_URIS = [f"http://localhost:{port}/callback" for port in CALLBACK_PORTS]


async def ensure_local_executor_oauth_app(owner_id: str) -> str:
    """Register the companion's public OAuth app unless one with its client
    id exists: "created", or "existing" for one left as it is."""
    client_id = ChatConfig().local_pc_executor_oauth_client_id
    app_id = str(uuid.uuid4())
    # A public client never presents a secret; the columns hold a random
    # placeholder, as oauth-tool writes for one.
    secret_hash, secret_salt = APIKeySmith().hash_key(
        f"agpt_secret_{secrets.token_urlsafe(32)}"
    )
    app = await OAuthApplication.prisma().upsert(
        where={"clientId": client_id},
        data={
            "create": {
                "id": app_id,
                "name": "AutoGPT Local Executor",
                "description": (
                    "The Local PC companion (autogpt-shim), which links a "
                    "computer to AutoGPT."
                ),
                "clientId": client_id,
                "clientSecret": secret_hash,
                "clientSecretSalt": secret_salt,
                "redirectUris": REDIRECT_URIS,
                "grantTypes": ["authorization_code", "refresh_token"],
                "scopes": [APIKeyPermission.CONNECT_LOCAL_EXECUTOR],
                "ownerId": owner_id,
                "isActive": True,
                "isPublic": True,
            },
            "update": {},
        },
    )
    return "created" if app.id == app_id else "existing"
