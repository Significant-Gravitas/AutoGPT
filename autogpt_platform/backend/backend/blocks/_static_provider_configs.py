"""Provider descriptions for services that don't yet have their own ``_config.py``.

Every provider in ``_STATIC_PROVIDER_CONFIGS`` below is declared here because
its block code currently lives either in a single shared file (e.g. the 8 LLM
providers in ``blocks/llm.py``) or in a single-file block that has no dedicated
directory (e.g. ``blocks/reddit.py``).

This file gets loaded by the block auto-loader in ``blocks/__init__.py``
(``rglob("*.py")`` picks it up) so the ``ProviderBuilder(...).build()`` calls
run at startup and populate ``AutoRegistry`` before the first API request.

**Migration path:** when a provider graduates into its own directory with a
proper ``_config.py`` (following the SDK pattern, e.g. ``blocks/linear/_config.py``),
delete its entry here. The metadata will still be served by
``GET /integrations/providers`` — it just moves to live next to the provider's
auth and webhook config.
"""

from backend.data.model import CredentialsType
from backend.sdk import ProviderBuilder

_STATIC_PROVIDER_CONFIGS: dict[str, tuple[str, tuple[CredentialsType, ...]]] = {
    # LLM providers that share blocks/llm.py
    "aiml_api": ("Unified access to 100+ AI models", ("api_key",)),
    "anthropic": ("Claude language models", ("api_key",)),
    "groq": ("Fast LLM inference", ("api_key",)),
    "llama_api": ("Llama model hosting", ("api_key",)),
    "ollama": ("Run open-source LLMs locally", ("api_key",)),
    "open_router": ("One API for every LLM", ("api_key",)),
    "openai": ("GPT models and embeddings", ("api_key",)),
    "v0": ("AI-generated UI components", ("api_key",)),
    # Single-file providers (one provider per standalone blocks/*.py file)
    "d_id": ("AI avatar and video generation", ("api_key",)),
    "e2b": ("Sandboxed code execution", ("api_key",)),
    "google_maps": ("Places, directions, geocoding", ("api_key",)),
    "http": ("Generic HTTP requests", ("api_key", "host_scoped")),
    "ideogram": ("Text-to-image generation", ("api_key",)),
    "medium": ("Publish stories and posts", ("api_key",)),
    "mem0": ("Long-term memory for agents", ("api_key",)),
    "openweathermap": ("Weather data and forecasts", ("api_key",)),
    "pinecone": ("Managed vector database", ("api_key",)),
    "reddit": ("Subreddits, posts, and comments", ("oauth2",)),
    "revid": ("AI-generated short-form video", ("api_key",)),
    "screenshotone": ("Automated website screenshots", ("api_key",)),
    "smtp": ("Send email via SMTP", ("user_password",)),
    "unreal_speech": ("Low-cost text-to-speech", ("api_key",)),
    "webshare_proxy": ("Rotating proxies for scraping", ("api_key",)),
}

for _name, (_description, _auth_types) in _STATIC_PROVIDER_CONFIGS.items():
    (
        ProviderBuilder(_name)
        .with_description(_description)
        .with_supported_auth_types(*_auth_types)
        .build()
    )
