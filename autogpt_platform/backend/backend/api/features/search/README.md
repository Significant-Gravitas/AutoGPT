# search

Shared search infrastructure for the platform. Anything that searches or
embeds *user/store/library/workspace/chat* content imports from here.

## Modules

- **`embeddings.py`** — generic embedding service. OpenAI wrapper
  (`generate_embedding`, `embed_query`), `UnifiedContentEmbedding` CRUD
  (`store_content_embedding`, `get_content_embedding`,
  `ensure_content_embedding`, `delete_content_embedding`),
  `semantic_search`, and admin ops (`backfill_all_content_types`,
  `cleanup_orphaned_embeddings`, `get_embedding_stats`).
- **`hybrid_search.py`** — `unified_hybrid_search` engine + BM25
  reranking (`tokenize`, `bm25_rerank`). Searches the
  `UnifiedContentEmbedding` table across every content type with
  combined semantic + lexical + recency + category scoring.
- **`content_handlers.py`** — pluggable per-`ContentType` handlers
  (store agent, block, doc, library agent, workspace file, chat
  session). Used by the embedding backfill/cleanup loops.
- **`service.py` / `routes.py`** — the `/api/search/global` endpoint
  used by the global search modal.

## Where things live

| Concern | Lives in |
|---|---|
| Generic engine (above) | `search/` |
| Store-listing wrappers (`store_embedding`, `hybrid_search` over store agents) | `store/embeddings.py`, `store/hybrid_search.py` |
| Per-feature background indexing | `library/embeddings.py`, `workspace/embeddings.py`, `copilot/chat_session_embeddings.py` |

**Rule:** anything used by more than one feature lives here. Feature-
specific wrappers (a SQL join that pulls store-agent metadata, the
fire-and-forget task that embeds a `LibraryAgent` row) stay in the
feature.

## Adding a new searchable content type

1. Add the enum value to `ContentType` (Prisma schema).
2. Add a `ContentHandler` subclass in `content_handlers.py` and
   register it in `CONTENT_HANDLERS`.
3. In the owning feature, add a small background task that calls
   `ensure_content_embedding` on write — see `library/embeddings.py`
   for the pattern.
4. The new type is now picked up by `unified_hybrid_search` and the
   global search endpoint automatically.
