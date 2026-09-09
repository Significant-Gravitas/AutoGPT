"""Microsoft Teams-specific configuration.

One bot registration serves every tenant conversation. Unlike Slack there is
no per-workspace install token: Teams routes everything through the Bot
Connector, and the bot authenticates itself with an Entra app credential.

Three secrets are required, not two. This adapter mints its outbound token
with a client-credentials grant against a *tenant* authority —
``https://login.microsoftonline.com/{TENANT_ID}/...`` — so the tenant ID is
part of the credential rather than an optional extra, and omitting it is the
single most common cause of 401s from the Connector.

That requirement is about the Azure **Bot** resource, which since 2025-07-31
must be created as Single Tenant or with a user-assigned managed identity.
The Entra **app registration** behind it has its own, separate audience
setting: this adapter does not require that to be single-tenant, so do not
narrow an existing registration on account of this note — doing so would cut
off any other integration that relies on it serving external tenants.

Registration checklist (once per environment — dev and prod need separate
registrations because a bot has exactly ONE messaging endpoint):
- Create the bot (either an Azure Bot resource, or ``teams app create`` from
  the Teams Developer CLI for a Teams-managed registration that needs no
  Azure subscription).
- Point its messaging endpoint at
  ``<PLATFORM_BASE_URL>/api/copilot-webhooks/teams/messages``.
- Enable the Microsoft Teams channel.
- Sideload the app package (see ``manifest.json``) into the tenant; the
  tenant must have custom app upload enabled.
"""

from backend.util.settings import AppEnvironment, Settings


def get_app_id() -> str:
    """The bot's Entra application (client) ID.

    Doubles as the expected ``aud`` claim on inbound Connector tokens and as
    ``bots[0].botId`` in the Teams app manifest — the manifest and this value
    must agree or Teams silently routes nothing.
    """
    return Settings().secrets.microsoft_client_id


def get_app_password() -> str:
    """The Entra client secret used to mint outbound Connector tokens."""
    return Settings().secrets.microsoft_client_secret


def get_tenant_id() -> str:
    """The tenant the single-tenant bot registration belongs to."""
    return Settings().secrets.microsoft_tenant_id


def is_configured() -> bool:
    """Whether the adapter should be mounted.

    All three credentials, or the local Playground bypass — which needs no
    credentials because nothing is authenticated in that mode.

    The adapter factory (``build_webhook_adapters``) calls this directly; the
    copilot tool's availability gate mirrors the credential check WITHOUT the
    bypass, because outbound posting must mint a real Connector token — see
    ``chat_platform._any_chat_platform_configured``.
    """
    if allow_unverified_requests():
        return True
    return bool(get_app_id() and get_app_password() and get_tenant_id())


def allow_unverified_requests() -> bool:
    """Local-dev escape hatch for the Microsoft 365 Agents Playground.

    The Playground simulates Teams without a tenant, tunnel or registration,
    but it sends activities with **no Bot Connector token** — so accepting its
    traffic means accepting unauthenticated requests on a public route.

    Guarded twice on purpose: the opt-in flag alone is not enough, the process
    must also be running as ``app_env=local``. A deployed environment is
    ``dev``/``prod``, so the bypass cannot activate there even if the flag is
    set by mistake.
    """
    settings = Settings()
    return (
        settings.config.app_env == AppEnvironment.LOCAL
        and settings.config.autopilot_bot_teams_allow_unverified
    )


# Teams imposes no documented per-message character limit; the real ceiling is
# a ~100KB UTF-16 payload for the whole activity. Our contract is
# character-based, so this is a deliberate choice well inside that budget:
# 8000 chars is 16KB of UTF-16 text, leaving ample room for activity JSON
# overhead, and keeps single messages readable in the Teams client.
MAX_MESSAGE_LENGTH = 8000

# Flush streamed replies a little under the cap so the boundary splitter can
# reach a natural break without overshooting.
CHUNK_FLUSH_AT = 7000

# Inbound-only ceiling: how much of a user-uploaded file we will download and
# ingest into the workspace.
MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024

# Outbound ceiling. A bot message activity is capped at ~28KB TOTAL, and the
# only single-shot file delivery is a base64 data URI inside that activity
# (~1.37x inflation + JSON overhead), so only genuinely small images can be
# inlined. Anything larger must go out as a link.
INLINE_IMAGE_BYTES = 16 * 1024

# Teams threads have no titles. The shared thread-naming logic still clamps
# candidate names against this, so any sane value satisfies the contract.
MAX_THREAD_NAME_LENGTH = 128

# A Teams typing indicator expires after ~3s; refresh just inside that.
TYPING_REFRESH_SECONDS = 2.0

# First delivery wins, per activity id. Covers Connector redeliveries and —
# because the inbound token does not sign the body — replays of a captured
# request, which stay valid for the token's ~1h lifetime plus clock skew.
ACTIVITY_DEDUPE_TTL_SECONDS = 75 * 60

# Inbound activities are signed by the Bot Connector, whose keys are published
# here. Discovery document -> jwks_uri -> signing keys.
OPENID_METADATA_URL = (
    "https://login.botframework.com/v1/.well-known/openidconfiguration"
)

# Every inbound Connector token must carry exactly this issuer.
TOKEN_ISSUER = "https://api.botframework.com"

# Outbound tokens are minted for this audience/scope.
CONNECTOR_SCOPE = "https://api.botframework.com/.default"

# ``serviceUrl`` arrives inside the (authenticated) activity and is where we
# send replies — i.e. it is attacker-influenced data that we then attach a
# bearer token to. Only HTTPS to these Connector hosts is ever dialled. Kept
# deliberately narrow: broad zones like ``.trafficmanager.net`` are
# customer-registerable (any Azure account can mint <name>.trafficmanager.net),
# which would defeat the point of the allowlist.
ALLOWED_SERVICE_URL_HOSTS = ("smba.trafficmanager.net",)
ALLOWED_SERVICE_URL_SUFFIXES = (
    ".botframework.com",
    ".botframework.azure.us",
)
