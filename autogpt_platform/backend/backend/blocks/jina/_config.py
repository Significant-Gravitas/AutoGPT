"""Provider registration for Jina — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

jina = ProviderBuilder("jina").with_description("Embeddings and reranking").build()
