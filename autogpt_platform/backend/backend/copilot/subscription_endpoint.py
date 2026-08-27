"""Running a turn on a user's own subscription, over an OpenAI-compatible API.

For a provider whose endpoint speaks the wire protocol the baseline path
already speaks, the whole runtime is this small object plus whatever
acquires the token: only the destination and the payer change.

That is fewer providers than it first looks, and the reason is worth
recording so the next reader does not assume otherwise:

- **Codex** -- ChatGPT's backend is not an OpenAI-compatible API, so that
  path drives a CLI that speaks it.
- **GitHub Copilot** -- there is no documented chat-completions endpoint. Its
  sanctioned path is a `copilot` runtime process spoken to over JSON-RPC,
  and the CLI *is* the API. (An undocumented HTTP endpoint does exist and
  does accept a user token, but it can change without notice and GitHub's
  own client code has a "client not supported" status for callers it
  rejects.)
- **Microsoft 365 Copilot** -- Graph conversation resources with a
  server-side conversation id, and cumulative SSE snapshots rather than
  deltas. Not this shape either.

So this is the seam for the providers that fit it, not the universal answer
for "linked subscription". A provider that needs a runtime gets a runtime;
what it must not get is a shared client.

**The whole point of this module is that a subscription client is never
shared.** The baseline path's normal client is a process-global singleton
built once from deployment config, which is right for a deployment key and
catastrophic for a user's: a cached client carries the token it was built
with, so one user's subscription would silently pay for every turn that
followed it in the same worker. Every function here builds a fresh client
and none of them writes to a global.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, SecretStr

# Same client class the baseline path uses, so tracing behaves identically.
from langfuse.openai import (  # pyright: ignore[reportPrivateImportUsage]
    AsyncOpenAI as LangfuseAsyncOpenAI,
)


class SubscriptionEndpoint(BaseModel):
    """Where a turn on a linked subscription goes, and what pays for it.

    Frozen and built per turn. Holds the token as a ``SecretStr`` so it does
    not surface in a log line or an exception repr -- these reach the same
    error paths as everything else, and a subscription token in a Sentry
    payload is the user's account, not ours.
    """

    model_config = ConfigDict(frozen=True)

    # Which provider this is, for cost attribution and error messages.
    auth_provider: str
    # OpenAI-compatible base URL for the provider's chat endpoint.
    base_url: str
    # The user's own token. Short-lived for some providers, which is why
    # this is resolved per turn rather than cached with the credential.
    api_key: SecretStr
    # Provider-side model slug for this turn, already resolved from the
    # tier. Carried here so the caller does not have to ask twice.
    model: str


def build_subscription_client(
    endpoint: SubscriptionEndpoint,
    *,
    timeout_seconds: float | None = None,
) -> LangfuseAsyncOpenAI:
    """A client bound to one user's subscription, for one turn.

    Deliberately not memoised, and deliberately not assigned to a module
    global. If this ever grows a cache, the cache key has to be the token
    itself and the entry has to be dropped when the turn ends -- at which
    point it is cheaper and safer to keep building one.
    """
    kwargs: dict = {
        "api_key": endpoint.api_key.get_secret_value(),
        "base_url": endpoint.base_url,
    }
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    return LangfuseAsyncOpenAI(**kwargs)


class SubscriptionEndpointUnavailable(Exception):
    """A turn is routed to a subscription this build cannot reach.

    Raised rather than answered with ``None`` on purpose. ``None`` means
    "the deployment's own key", and returning it here would run a turn the
    user routed to their own account on ours -- billing us, and telling the
    user their subscription was used when it was not. A turn that cannot go
    where it was routed must fail.
    """


# Resolvers keyed by chat provider, registered by each provider's own module.
#
# The table in ``subscription_providers`` says a provider exists and what it
# is called. This says how to actually reach it, and stays a registry rather
# than a field on the profile because a resolver needs the network, the
# credential store, and the user -- none of which belong in a constants
# table that must stay importable without them.
_RESOLVERS: dict[str, "SubscriptionEndpointResolver"] = {}


class SubscriptionEndpointResolver(Protocol):
    async def __call__(
        self,
        *,
        user_id: str,
        credential_id: str,
        model: str,
    ) -> SubscriptionEndpoint: ...


def register_endpoint_resolver(
    auth_provider: str, resolver: SubscriptionEndpointResolver
) -> None:
    _RESOLVERS[auth_provider] = resolver


async def resolve_subscription_endpoint(
    auth_provider: str | None,
    credential_id: str | None,
    *,
    user_id: str | None,
    model: str,
) -> SubscriptionEndpoint | None:
    """Where this turn goes, or ``None`` for the deployment's own key.

    ``None`` is returned only for the platform route -- an actual absence of
    a subscription, not a failure to find one. Every other outcome that is
    not a working endpoint raises, because the caller's fallback is the
    deployment key and silently taking it is the one wrong answer here.
    """
    if auth_provider is None or auth_provider == "platform":
        return None
    if credential_id is None:
        raise SubscriptionEndpointUnavailable(
            f"{auth_provider} route carries no credential to run on"
        )
    if user_id is None:
        raise SubscriptionEndpointUnavailable(
            f"{auth_provider} route has no user to resolve a credential for"
        )
    resolver = _RESOLVERS.get(auth_provider)
    if resolver is None:
        raise SubscriptionEndpointUnavailable(
            f"No runtime for {auth_provider} in this build"
        )
    return await resolver(user_id=user_id, credential_id=credential_id, model=model)
