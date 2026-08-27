"""Per-provider behaviour table for bring-your-own-subscription chat routes.

A chat can run on the deployment's own key (``platform``) or on a subscription
the user already pays for and has linked. Each linked provider needs the same
dozen answers -- what to call it, what backs a run, which entitlement gates it,
which catalog surface resolves its models, what a failure from it means -- and
until now those answers lived as ``if auth_provider == "codex"`` branches
spread across transports, offers, tiers, the model router and the executor.

That works for one provider and quietly rots at two: every branch is a place
the second provider can be forgotten, and the failure mode is silent (a
provider that reads as the other one, an offer with the wrong plan named on
it) rather than a crash.

This is the same shape as ``TransportProfile`` in ``config.py``, for the same
reason its docstring gives: adding a provider should be "add a row", not "find
every branch". Where a value is genuinely presentation, it lives here as data.
Where behaviour really does differ -- acquiring a credential, standing up a
runtime -- the profile names the strategy and the implementation stays in the
provider's own module.

Deliberately *not* here: anything that needs a user id. Entitlement checks,
credential lookups and availability are per-user questions asked at call time;
this table is the static description of a provider, so it stays importable
without touching Redis or the database.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.copilot.config import CopilotLlmAuthProvider
from backend.integrations.providers import ProviderName
from backend.util.entitlements import Entitlement

# How a user's credential for this provider is obtained.
#
# ``platform``  -- there is no credential; the deployment's own key runs the
#                  turn. Only the built-in platform route uses this.
# ``runtime_device_code`` -- a provider CLI owns the device-code dance and the
#                  refresh; we drive that CLI and read the tokens it wrote.
#                  Chosen when the provider does not offer us a first-party
#                  OAuth app (Codex).
# ``oauth_app``  -- we register our own OAuth application with the provider and
#                  run a standard authorization-code or device flow ourselves,
#                  holding and refreshing the token like any other integration.
CredentialStrategy = Literal["platform", "runtime_device_code", "oauth_app"]


class SubscriptionProviderProfile(BaseModel):
    """One row per chat provider. Frozen: this is a constants table."""

    model_config = ConfigDict(frozen=True)

    key: CopilotLlmAuthProvider

    # --- identity and presentation -------------------------------------
    # What a person calls it. Shown on the connection chip, the picker row,
    # and the Settings entry.
    display_name: str
    # Presentation grouping, deliberately not the credential discriminator:
    # a ChatGPT subscription and an OpenAI API key are one family to a reader
    # and two entirely different credentials to the server.
    provider_family: str
    # The string the frontend keys its connect affordances off. Stays a
    # provider-specific token rather than a boolean so a client can tell
    # "linked account" apart from "which linked account".
    auth_method: str
    # "Your ChatGPT plan" -- what pays for a run, in the user's terms.
    backed_by_label: str
    # One sentence for the picker row: what a new chat on this connection is
    # backed by, and what it does or does not spend.
    description: str
    # Real edges a user can hit on this route, not policy notes. Rendered as
    # a bulleted list under the connection.
    limitations: tuple[str, ...] = ()

    # --- gating ---------------------------------------------------------
    # Entitlement required to use the provider at all. ``None`` means the
    # route is open to anyone who can link an account.
    entitlement: Entitlement | None = None
    # Shown when the entitlement is missing, with somewhere to go about it.
    lock_reason: str | None = None
    unlock_href: str | None = None

    # --- credentials ----------------------------------------------------
    credential_strategy: CredentialStrategy
    # The credential row's provider, when there is one. ``None`` for the
    # platform route, which has no credential.
    credential_provider: ProviderName | None = None
    # A provider whose sign-in hands control to a browser and a device code
    # needs far longer than a normal API call before the proxy gives up.
    login_timeout_seconds: int | None = None

    # --- connect flow copy ----------------------------------------------
    # What the button that starts the sign-in says. The provider's own name
    # for the act, so "Sign in with ChatGPT" rather than a generic "Connect".
    connect_button_label: str | None = None
    # Whose terms the user is agreeing to when they press it. Named because
    # sending someone to a third party's login without saying whose is the
    # kind of thing that erodes trust quietly.
    terms_company: str | None = None
    # The provider slug the integrations UI groups this credential under. A
    # ChatGPT subscription files under "openai" beside an OpenAI API key,
    # because that is where a person looks for it -- even though the two are
    # entirely different credentials to the server.
    display_alias: str | None = None

    # --- routing --------------------------------------------------------
    # Registry surface that resolves (mode, tier) -> model for this provider.
    # ``None`` means the platform router decides.
    route_surface: str | None = None
    # If set, a model is only offered on this provider when the catalog says
    # it comes from this vendor -- a ChatGPT plan cannot serve Gemini.
    catalog_vendor: str | None = None


PLATFORM = SubscriptionProviderProfile(
    key="platform",
    display_name="AutoGPT Platform",
    provider_family="autogpt",
    auth_method="deployment",
    backed_by_label="Your AutoGPT plan",
    description="New chats are backed by your AutoGPT plan, and spend AutoGPT credits.",
    credential_strategy="platform",
)

CODEX = SubscriptionProviderProfile(
    key="codex",
    display_name="ChatGPT",
    provider_family="openai",
    auth_method="chatgpt_oauth",
    backed_by_label="Your ChatGPT plan",
    description=(
        "New chats are backed by your ChatGPT plan, and spend no AutoGPT credits."
    ),
    # Stated because it is a real edge a user can hit, not a policy note: the
    # builder panel rejects a codex route outright.
    limitations=("The agent builder's chat panel always runs on AutoGPT.",),
    entitlement=Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
    lock_reason="A Max plan or higher is required to use ChatGPT.",
    unlock_href="/settings/billing",
    credential_strategy="runtime_device_code",
    credential_provider=ProviderName.CODEX,
    login_timeout_seconds=15 * 60,
    connect_button_label="Sign in with ChatGPT",
    terms_company="OpenAI",
    display_alias="openai",
    route_surface="copilot_codex",
    catalog_vendor="openai",
)

GITHUB_COPILOT = SubscriptionProviderProfile(
    key="github_copilot",
    display_name="GitHub Copilot",
    provider_family="github",
    auth_method="github_oauth",
    backed_by_label="Your GitHub Copilot subscription",
    description=(
        "New chats are backed by your GitHub Copilot subscription, and spend "
        "no AutoGPT credits."
    ),
    # Copilot meters premium requests against the user's own allowance, so a
    # long agent run costs them something even though it costs us nothing.
    # Said plainly here rather than discovered on a bill.
    limitations=(
        "The agent builder's chat panel always runs on AutoGPT.",
        "Runs count against your Copilot premium request allowance.",
    ),
    entitlement=Entitlement.GITHUB_COPILOT_SUBSCRIPTION_TRANSPORT,
    lock_reason="A Max plan or higher is required to use GitHub Copilot.",
    unlock_href="/settings/billing",
    # GitHub gives us a real OAuth application of our own, so unlike Codex we
    # hold and refresh the token ourselves rather than driving a CLI's login
    # and reading what it wrote.
    credential_strategy="oauth_app",
    credential_provider=ProviderName.GITHUB_COPILOT,
    login_timeout_seconds=15 * 60,
    connect_button_label="Sign in with GitHub",
    terms_company="GitHub",
    display_alias="github",
    route_surface="copilot_github",
    catalog_vendor=None,
)

_PROFILES: dict[str, SubscriptionProviderProfile] = {
    profile.key: profile for profile in (PLATFORM, CODEX, GITHUB_COPILOT)
}


def profile_for(auth_provider: str) -> SubscriptionProviderProfile:
    """The profile for a provider key.

    Raises rather than falling back to the platform row: a provider we cannot
    describe is a bug in this table, and answering with the wrong plan's name
    is worse than failing loudly.
    """
    try:
        return _PROFILES[auth_provider]
    except KeyError:
        raise ValueError(f"No subscription provider profile for {auth_provider!r}")


def known_profiles() -> tuple[SubscriptionProviderProfile, ...]:
    """Every provider, platform first, then linked accounts in table order."""
    return tuple(_PROFILES.values())


def linked_profiles() -> tuple[SubscriptionProviderProfile, ...]:
    """Providers a user links an account to -- everything but the platform."""
    return tuple(p for p in _PROFILES.values() if p.credential_strategy != "platform")
