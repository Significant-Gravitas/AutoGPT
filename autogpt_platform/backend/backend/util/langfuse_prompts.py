"""The Langfuse client our managed prompts are fetched with.

Langfuse only serves prompts now: the AutoPilot system prompt
(``copilot/service.py``) and the onboarding greeting
(``api/features/onboarding_dump/intro.py``). Conversations are traced in
PostHog (``copilot/ai_spans.py`` and OpenRouter Broadcast), so this client is
built with tracing off: it sets up no OpenTelemetry tracer and sends no spans,
whatever ``LANGFUSE_TRACING_ENABLED`` says.

The SDK keeps one set of resources per public key, and the first client built
for a key decides whether it traces, so nothing else in the backend may build a
Langfuse client or call ``langfuse.get_client()``.
"""

import functools

from langfuse import Langfuse

from backend.util.settings import Settings


def langfuse_prompts_configured() -> bool:
    """Whether Langfuse credentials are set, so prompts can be fetched."""
    secrets = Settings().secrets
    return bool(secrets.langfuse_public_key and secrets.langfuse_secret_key)


@functools.cache
def get_langfuse_prompt_client() -> Langfuse:
    """The process's Langfuse client, for ``get_prompt`` only."""
    # The configured credentials, which ``.env`` puts in ``Settings`` but not
    # in the environment the SDK would otherwise read them from. ``host``
    # still yields to ``LANGFUSE_BASE_URL``, as it would by default.
    secrets = Settings().secrets
    return Langfuse(
        public_key=secrets.langfuse_public_key or None,
        secret_key=secrets.langfuse_secret_key or None,
        host=secrets.langfuse_host or None,
        tracing_enabled=False,
    )
