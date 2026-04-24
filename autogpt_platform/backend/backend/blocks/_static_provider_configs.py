"""Provider descriptions for services that don't yet have their own ``_config.py``.

Every provider in ``_STATIC_PROVIDER_DESCRIPTIONS`` below is declared here
because its block code currently lives either in a single shared file
(e.g. the 7 LLM providers in ``blocks/llm.py``) or in a single-file block
that has no dedicated directory (e.g. ``blocks/reddit.py``).

This file gets loaded by the block auto-loader in ``blocks/__init__.py``
(``rglob("*.py")`` picks it up) so the ``ProviderBuilder(...).build()`` calls
run at startup and populate ``AutoRegistry`` before the first API request.

**Migration path:** when a provider graduates into its own directory with a
proper ``_config.py`` (following the SDK pattern, e.g. ``blocks/linear/_config.py``),
delete its entry here. The description will still be served by
``GET /integrations/providers`` — it just moves to live next to the provider's
auth and webhook config.
"""

from backend.sdk import ProviderBuilder

_STATIC_PROVIDER_DESCRIPTIONS: dict[str, str] = {
    # LLM providers that share blocks/llm.py
    "aiml_api": "Unified access to 100+ AI models",
    "anthropic": "Claude language models",
    "groq": "Fast LLM inference",
    "llama_api": "Llama model hosting",
    "ollama": "Run open-source LLMs locally",
    "open_router": "One API for every LLM",
    "openai": "GPT models and embeddings",
    "v0": "AI-generated UI components",
    # Single-file providers (one provider per standalone blocks/*.py file)
    "d_id": "AI avatar and video generation",
    "database": "Internal data storage",
    "e2b": "Sandboxed code execution",
    "google_maps": "Places, directions, geocoding",
    "http": "Generic HTTP requests",
    "ideogram": "Text-to-image generation",
    "medium": "Publish stories and posts",
    "mem0": "Long-term memory for agents",
    "openweathermap": "Weather data and forecasts",
    "pinecone": "Managed vector database",
    "reddit": "Subreddits, posts, and comments",
    "revid": "AI-generated short-form video",
    "screenshotone": "Automated website screenshots",
    "smtp": "Send email via SMTP",
    "unreal_speech": "Low-cost text-to-speech",
    "webshare_proxy": "Rotating proxies for scraping",
}

for _name, _description in _STATIC_PROVIDER_DESCRIPTIONS.items():
    ProviderBuilder(_name).with_description(_description).build()
