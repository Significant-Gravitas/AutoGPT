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

import os

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

    # --- availability ---------------------------------------------------
    # Environment variable an operator must set truthy before this provider
    # is offered at all. ``None`` means "always available".
    #
    # This exists for a provider whose sign-in works but is not ours to use:
    # where the vendor publishes no third-party client registration and the
    # only route is a first-party client id, running it is a decision about
    # the deployment's own relationship with that vendor. That belongs to
    # whoever operates the deployment, not to a default in this file.
    opt_in_env: str | None = None
    # Whether this build can actually complete the round trip: acquire a
    # credential and run a turn on it. Both halves, because a provider with
    # only one is useless -- a sign-in that leads to chats that all fail is
    # worse than no sign-in.
    #
    # Separate from ``opt_in_env``, which is the operator declining something
    # that works. This is us not having built it yet, and it is a row in the
    # table rather than an unwritten row so that everything downstream --
    # the offers, the gates, the connect copy, the tests -- is already
    # written and exercised when the runtime lands. What flips is this flag.
    runtime_ready: bool = True

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
    # How this connection reads in the aliased entry's one-line summary, so
    # the OpenAI card can say "OpenAI models via API key or your ChatGPT
    # subscription" without the client knowing that codex and openai are the
    # same card. A noun phrase, joined with "or" by whoever composes it.
    connection_summary: str | None = None

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
    connection_summary="your ChatGPT subscription",
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
    # No OAuth handler and no runtime yet, so it is described and tested but
    # not offered: the connect button would open a login route that does not
    # exist.
    runtime_ready=False,
    credential_strategy="oauth_app",
    credential_provider=ProviderName.GITHUB_COPILOT,
    login_timeout_seconds=15 * 60,
    connect_button_label="Sign in with GitHub",
    terms_company="GitHub",
    # Its own card, unlike ChatGPT under OpenAI. GitHub already offers OAuth
    # for repositories, and a card shows one tab per auth method -- filing
    # Copilot under it would put two different OAuth sign-ins on one tab and
    # leave whichever lost unreachable. It is also a product a person looks
    # for by name.
    display_alias=None,
    route_surface="copilot_github",
    catalog_vendor=None,
)

GROK = SubscriptionProviderProfile(
    key="grok",
    display_name="Grok",
    provider_family="xai",
    auth_method="grok_oauth",
    backed_by_label="Your SuperGrok subscription",
    description=(
        "New chats are backed by your SuperGrok or X Premium+ subscription, "
        "and spend no AutoGPT credits."
    ),
    limitations=(
        "The agent builder's chat panel always runs on AutoGPT.",
        "xAI has been observed refusing inference on lower SuperGrok tiers.",
    ),
    entitlement=Entitlement.GROK_SUBSCRIPTION_TRANSPORT,
    lock_reason="A Max plan or higher is required to use Grok.",
    unlock_href="/settings/billing",
    # Off unless an operator turns it on, and the only provider here that is.
    #
    # xAI ships a working device-code flow, and several editors use it, but
    # they publish no way to register a client of our own -- the only route
    # is xAI's first-party client id, which their own source obfuscates. That
    # makes every third-party sign-in an impersonation of their CLI, against
    # an AUP that forbids "bypassing our systems or protective measures", and
    # xAI can revoke it for everyone with one allowlist change.
    #
    # It is a real capability and some deployments will want it. It is not a
    # default we can make on an operator's behalf.
    opt_in_env="CHAT_ENABLE_GROK_SUBSCRIPTION",
    runtime_ready=False,
    credential_strategy="runtime_device_code",
    credential_provider=ProviderName.GROK,
    login_timeout_seconds=15 * 60,
    connect_button_label="Sign in with Grok",
    terms_company="xAI",
    display_alias="xai",
    connection_summary="your SuperGrok subscription",
    route_surface="copilot_grok",
    catalog_vendor="xai",
)

_PROFILES: dict[str, SubscriptionProviderProfile] = {
    profile.key: profile for profile in (PLATFORM, CODEX, GITHUB_COPILOT, GROK)
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


def is_enabled(profile: SubscriptionProviderProfile) -> bool:
    """Whether this deployment offers the provider at all.

    Two reasons it may not: this build cannot do it yet, or the operator has
    declined something it can. Both end in the same place -- the provider is
    absent from every list -- so both are answered here rather than at each
    of the dozen call sites that would otherwise have to remember.

    Read at call time rather than at import so an operator's choice takes
    effect on restart without the table caching a stale answer.
    """
    if not profile.runtime_ready:
        return False
    if profile.opt_in_env is None:
        return True
    return os.getenv(profile.opt_in_env, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def linked_profiles() -> tuple[SubscriptionProviderProfile, ...]:
    """Providers a user can link an account to, on this deployment.

    Excludes the platform route, which has no account to link, and anything
    the operator has not opted into.
    """
    return tuple(
        p
        for p in _PROFILES.values()
        if p.credential_strategy != "platform" and is_enabled(p)
    )


def profile_for_credential_provider(
    credential_provider: str,
) -> SubscriptionProviderProfile | None:
    """The profile that owns a credential provider, or ``None``.

    ``profile_for`` is keyed by the chat route ("codex"); this is keyed by the
    credential row's provider, which is what code holding a credential has.
    They happen to be the same string today for every provider, and nothing
    guarantees they stay that way -- so the lookup is explicit rather than a
    cast.
    """
    for profile in _PROFILES.values():
        if profile.credential_provider == credential_provider:
            return profile
    return None


def is_subscription_credential(credential_provider: str) -> bool:
    """Whether a credential is a linked subscription rather than a key.

    The distinction matters wherever money does: a run on one of these is
    paid for by the user's own plan, so it is neither charged to AutoGPT
    credits nor blocked by AutoGPT's paywall.
    """
    return profile_for_credential_provider(credential_provider) is not None


def runtime_is_available(auth_provider: str) -> bool:
    """Whether a turn can be run on this connection in this build.

    Asked at dispatch, not only when a session is routed: a session outlives
    the deployment that created it, so an operator turning a provider off --
    or a build that no longer ships its runtime -- must stop the turn rather
    than let it fall through to whichever path is next.

    The platform route is always available; it is the deployment itself.
    """
    if auth_provider == "platform":
        return True
    try:
        return is_enabled(profile_for(auth_provider))
    except ValueError:
        return False
