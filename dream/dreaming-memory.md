# Dreaming Feature — Memory System Technical Findings

## 1. Graphiti Client Surface

Our wrapper (`backend/copilot/graphiti/client.py`) exposes only `get_graphiti_client(group_id)` and `evict_client(group_id)`. The returned object is an upstream `graphiti_core.Graphiti` instance at version `0.28.2` (`pyproject.toml:37`).

**Methods available on the upstream client (v0.28.2):**

- `add_episode(name, episode_body, source_description, reference_time, source, group_id, uuid, update_communities, entity_types, excluded_entity_types, previous_episode_uuids, edge_types, edge_type_map, custom_extraction_instructions, saga, saga_previous_episode_uuid)` — single episode ingestion with LLM entity extraction
- `add_episode_bulk(bulk_episodes, group_id, ...)` — batch ingestion; this exists upstream but is **not used anywhere** in our codebase
- `retrieve_episodes(reference_time, last_n, group_ids, source, driver, saga)` — returns `list[EpisodicNode]` ordered by recency; the only time-proximity fetch we have, limited to `last_n` from `reference_time`
- `search(query, center_node_uuid, group_ids, num_results, search_filter, driver)` — hybrid vector+fulltext search returning `RELATES_TO` edges
- `search_(query, config, group_ids, ...)` — advanced configurable search returning both nodes and edges
- `add_triplet(source_node, edge, target_node)` — manual fact insertion bypassing LLM extraction
- `get_nodes_and_edges_by_episode(episode_uuids)` — retrieves the subgraph for specific episodes
- `remove_episode(episode_uuid)` — hard-deletes an episode
- `build_communities(group_ids, driver)` — community clustering (unused in our code)
- `self.nodes` / `self.edges` — namespace objects for entity node and edge CRUD

**What we cannot do today that dreaming needs:**

- No `list_episodes_since(user_id, since_ts)` — `retrieve_episodes` only takes `last_n` from `reference_time`, not a `since` timestamp or time-window filter
- No `mark_superseded(uuid)` — the upstream client has no concept of memory status; the closest primitive is `_soft_delete_edges` in `graphiti_forget.py:268-280` which sets `invalid_at`/`expired_at`
- No `bulk_demote(uuids)` — the existing soft-delete is a serial Python loop, not a single bulk Cypher
- No `merge_episodes(uuid_a, uuid_b)` — no consolidation primitive exists anywhere
- `add_episode_bulk` exists upstream but is unwired in our ingest path; using it in a dream pass would bypass the per-user serialization queue

## 2. FalkorDB Layout

FalkorDB is used as a property-graph database accessed through `AutoGPTFalkorDriver` (`backend/copilot/graphiti/falkordb_driver.py`), which subclasses the upstream `FalkorDriver`.

**Node labels:**
- `:Entity` — named real-world entities (people, companies, projects); properties: `uuid`, `name`, `summary`, `group_id`, `labels`, `created_at`
- `:Episodic` — episode nodes; properties: `uuid`, `name`, `group_id`, `source_description`, `content`, `entity_edges` (list of edge uuids), `created_at`, `valid_at`, `source`
- `:Community` — entity cluster nodes (built by `build_communities`)
- `:Saga` — saga tracking nodes (unused in our code)

**Edge types:**
- `:RELATES_TO` — semantic fact edges between `:Entity` nodes; properties: `uuid`, `name`, `fact`, `fact_embedding`, `group_id`, `episodes`, `created_at`, `expired_at`, `valid_at`, `invalid_at`, `source_node_uuid`, `target_node_uuid`, `attributes`
- `:MENTIONS` — provenance edges from `:Episodic` to `:Entity`; properties: `uuid`, `group_id`, `created_at`
- `:HAS_MEMBER` — community membership edges
- `:HAS_EPISODE` / `:NEXT_EPISODE` — episode graph traversal

**Key insight for dreaming:**

(a) **Read recent episodes for a user** — `retrieve_episodes(reference_time=now, last_n=N, group_ids=[group_id])` or a raw Cypher query against `:Episodic` nodes filtered by `group_id` and `created_at > $since_ts`. The driver supports arbitrary Cypher via `driver.execute_query(query, **kwargs)` (`falkordb_driver.py` inherits this from upstream). The AGENTS.md query cookbook already shows a working pattern for this.

(b) **Write new consolidated facts without duplicating** — Use `client.add_episode(...)` with `EpisodeType.json` and a `MemoryEnvelope` envelope. Graphiti-core runs deduplication during extraction using the LLM, but this only checks for textual conflicts at extraction time, not pre-existing edge UUIDs. Dream-written episodes should use a distinct `name` prefix (e.g. `dream_{user_id}_{timestamp}`) to be identifiable later.

(c) **Mark old facts `superseded`** — The graph has no `status` property on edges at the Graphiti layer. The closest mechanism is `_soft_delete_edges` which sets `invalid_at = datetime()` and `expired_at = datetime()`. A Cypher `SET e.status = 'superseded'` would work as a custom property but would not be filtered by Graphiti's default search (which filters on `invalid_at IS NULL`). This is actually advantageous — a custom `status` property on `:RELATES_TO` edges survives round-trips, won't break existing queries, and can be used as an explicit dream-demote flag.

## 3. Scope Semantics

Scope is **not a FalkorDB label or separate sub-graph**. Each user gets exactly one FalkorDB database named `user_{sanitized_user_id}` (the `group_id`), and all scopes for that user live in the same database. Scope is a **property on the MemoryEnvelope JSON blob** stored as the `content` field of `:Episodic` nodes.

From `client.py:49-75`, `derive_group_id("abc-user-id")` → `"user_abc-user-id"`, and the `AutoGPTFalkorDriver` is instantiated with `database=group_id`. There is one client cache entry and one FalkorDB database per user, not per scope.

Scope filtering happens in Python at read time by parsing the episode JSON body and checking `data.get("scope", "real:global")` — see `graphiti_search.py:184-213` (`_filter_episodes_by_scope`) and `context.py:106-117` (`_is_non_global_scope`). The `client.search()` call returns cross-scope results; scope filtering is a post-processing step in our code only.

**Implication for dreaming:** A single dream pass per user operates over all scopes in the same FalkorDB database. If the dream pass should consolidate only within a scope, it must filter episodes in Python after fetching. Three separate runs per scope are not required by the infrastructure but may be desirable semantically. Anthropic's *Dreams* uses one memory store per pass, so a single run is the simpler default.

## 4. MemoryEnvelope Round-Trip

The `MemoryEnvelope` (`memory_model.py:83-118`) is serialized as JSON via `envelope.model_dump_json()` and stored as the `content` field of an `:Episodic` node when ingested with `EpisodeType.json`. Graphiti-core runs LLM entity extraction against the `content` field and creates `:Entity` nodes and `:RELATES_TO` edges from the extracted facts.

**What survives the round-trip:**

When episodes are retrieved via `retrieve_episodes` or raw Cypher, the full JSON blob is in `episode.content` (accessed via `extract_episode_body_raw` in `_format.py:21-33`). All `MemoryEnvelope` fields — `content`, `source_kind`, `scope`, `memory_kind`, `status`, `confidence`, `provenance`, `rule`, `procedure` — are present in that blob and can be `json.loads()`-parsed back to a dict.

**What does NOT survive on `:RELATES_TO` edges:**

The `RELATES_TO` edge schema has `uuid`, `name`, `fact`, `fact_embedding`, `group_id`, `episodes`, temporal validity fields, and `attributes`. There is no `status`, `confidence`, `source_kind`, `scope`, or `provenance` property on edges — these exist only on the `:Episodic` node's raw content. When `client.search()` returns edges, `extract_fact(e)` reads `e.fact` or `e.name` — there is no way to recover the originating envelope's `status` or `confidence` from an edge alone.

**Critical for dreaming:** The dream pass reads `MemoryEnvelope.status` to decide what to demote, but `status` is only in the `:Episodic` node's raw content blob. To flip `status=superseded` on a fact, the dream pass must: (1) find the `Episodic` node by uuid or name, (2) parse its `content` JSON, (3) mutate `status`, and (4) either write a new episode with the updated envelope, or execute a raw Cypher `SET ep.content = $new_json` on the `:Episodic` node. Neither path is encapsulated by any existing helper.

## 5. The Forget Path

`MemoryForgetSearchTool` (`graphiti_forget.py:26-135`) calls `client.search()` to find candidate `:RELATES_TO` edges and returns their UUIDs. `MemoryForgetConfirmTool` (`graphiti_forget.py:138-251`) then applies either:

- **Soft delete** (`_soft_delete_edges`, `graphiti_forget.py:254-290`): raw Cypher `SET e.invalid_at = datetime(), e.expired_at = datetime()` on `:MENTIONS|RELATES_TO|HAS_MEMBER` edges by uuid. This is temporal invalidation — edges are excluded from default `search()` results (which filter `invalid_at IS NULL`), but nodes and the `:Episodic` content blob remain. Reversible.
- **Hard delete** (`_hard_delete_edges`, `graphiti_forget.py:293-349`): raw Cypher `DELETE e` plus cleanup of `ep.entity_edges` back-references. Irreversible.

**Can we use this for dream-driven demotion?** The soft delete is the closest hook. However, setting `invalid_at` means the fact is treated as temporally ended, not semantically superseded — it simply vanishes from search results. A dream pass that wants to preserve history (for ratification scoring) but mark facts as superseded should use a custom `status` property instead. The raw driver is already used directly in `_soft_delete_edges`, so adding a `SET e.status = 'superseded'` to that Cypher is a one-line extension. No new mutation infrastructure is needed — just a new helper function alongside the existing ones.

## 6. Per-User Concurrency

`_ingestion_worker` in `ingest.py:62-99` is a single `asyncio.Task` per user per event loop, serializing all `add_episode()` calls through an `asyncio.Queue(maxsize=100)`. The queue is keyed by `user_id` in a `WeakKeyDictionary` scoped to the running asyncio event loop (`ingest.py:27-47`).

**Concurrent write conflict risk:** The scheduler runs as a separate process/container (`docker-compose.yml:82-86`) on its own asyncio loop. A dream pass running there would call `get_graphiti_client(group_id)` and `client.add_episode(...)` directly — not through the `enqueue_episode` queue, which lives in the CoPilot executor process. This means **a dream pass and a live chat session can simultaneously issue `add_episode()` calls** to the same FalkorDB database without the queue's serialization.

Graphiti-core `add_episode` does graph writes that involve reading existing nodes, running LLM deduplication, and then writing. Two concurrent calls to the same FalkorDB database could produce duplicate or conflicting entity extraction. FalkorDB itself is single-threaded (Redis-like), so individual Cypher queries are atomic, but the LLM extraction — which happens between graph reads and graph writes — is not protected by any lock in the dream-side call.

**Safe pattern:** The dream pass should either (a) use the same `enqueue_episode` queue by sending its consolidated episodes to the CoPilot executor via an inter-process mechanism (e.g. a dedicated RabbitMQ message), or (b) check whether a user has had any activity in the last N minutes before starting the pass and bail if so. The simplest safe guard is a user-level advisory lock using Redis SETNX keyed as `dream_lock:{group_id}`, released when the pass completes or times out.

## 7. Embeddings

Embeddings are generated by `OpenAIEmbedder` (`client.py:155-160`) using `graphiti_config.embedder_model` (default: `text-embedding-3-small`, `config.py:52`). The API key falls back through `CHAT_OPENAI_API_KEY` → `OPENAI_API_KEY` (`config.py:112-115`). This goes directly to OpenAI, bypassing OpenRouter.

Embeddings are created by graphiti-core during `add_episode()` — specifically, the `fact_embedding` on each extracted `:RELATES_TO` edge is embedded at extraction time using this embedder. When a dream pass calls `add_episode()` (or `enqueue_episode()`), the same embedder is invoked automatically. No manual embedding step is needed for dream-written episodes.

**Cost:** `text-embedding-3-small` is cheap (~$0.02/million tokens). A dream pass consolidating 50 episodes and writing back 10–20 new facts is likely in the $0.001 range per user per night. At scale (10k users), this is ~$10/night — negligible but worth instrumenting.

## 8. What's Missing for Dreaming

This is the actionable TODO list, ordered by priority.

**Priority 1 — Required before any dream pass can run:**

- `list_episodes_in_window(user_id, since_ts, until_ts)` — new async function in `ingest.py` or a new `backend/copilot/graphiti/fetch.py`. Uses raw driver Cypher: `MATCH (ep:Episodic {group_id: $group_id}) WHERE ep.created_at >= $since_ts AND ep.created_at < $until_ts RETURN ep ORDER BY ep.created_at`. Returns parsed `MemoryEnvelope` objects where the content is valid JSON, and raw strings otherwise.

- `list_active_facts(user_id, scope)` — same file. Cypher: `MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) WHERE e.group_id = $group_id AND e.invalid_at IS NULL AND (e.expired_at IS NULL OR e.expired_at > datetime()) RETURN e ORDER BY e.created_at DESC LIMIT 200`. This is the dream pass's input for consolidation.

**Priority 2 — Required to write dream outputs safely:**

- `enqueue_dream_episode(user_id, envelope)` — wrapper around existing `enqueue_episode()` that stamps `source_kind=assistant_derived`, `provenance="dream:{timestamp}"`, and `status=tentative`. No new infrastructure — just a thin function in `ingest.py` with correct defaults and a distinct episode name prefix `dream_{uuid4}`.

- `mark_edges_superseded(driver, uuids)` — new function alongside `_soft_delete_edges` in `graphiti_forget.py`. Cypher: `MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.status = 'superseded', e.expired_at = datetime()`. Setting `expired_at` retains the "soft" invalidation behavior so the edge falls out of Graphiti's default search while the `status` property provides a queryable audit trail. This is a 20-line addition to the existing forget module.

- Redis advisory lock helper: `async with dream_lock(group_id, redis_client, ttl=600)` — prevents dream/chat write conflicts. The lock acquirer is the scheduler process; the CoPilot executor processes continue using their per-user asyncio queues unaffected.

**Priority 3 — Required for ratification and metrics:**

- `list_dream_episodes(user_id, since_ts)` — filter `list_episodes_in_window` by `provenance.startswith("dream:")` in the parsed envelope. Used to score how many dream memories were confirmed by subsequent user activity.

- `ratify_episode(user_id, episode_uuid)` — Cypher `SET ep.status = 'active'` on the `:Episodic` node's content JSON (requires re-serialization). Alternatively, write a new episode with `status=active` and soft-delete the tentative one. The latter is safer because it goes through normal ingestion and re-embeds the fact.

**Priority 4 — Optional quality improvements:**

- `merge_episodes(user_id, source_uuids, merged_content)` — hard-delete the source episodes after writing the merged one. This requires calling both `client.remove_episode(uuid)` for the sources and `enqueue_dream_episode` for the consolidated output. Risky without ratification — defer to v2.

- Expose `add_episode_bulk` through the ingestion queue by adding a `enqueue_bulk` variant that respects the per-user serialization order. Currently the bulk upstream method bypasses the queue entirely.

---

**What to build first (ordered):**

1. `list_episodes_in_window(user_id, since_ts, until_ts)` in `backend/copilot/graphiti/fetch.py` — this unblocks the dream pass's read side.
2. `mark_edges_superseded(driver, uuids)` in `backend/copilot/tools/graphiti_forget.py` — 20-line addition alongside `_soft_delete_edges`.
3. Redis advisory lock for `group_id` — prevents dream/chat write conflicts; live in `backend/copilot/graphiti/locks.py`.
4. `enqueue_dream_episode(user_id, envelope)` in `backend/copilot/graphiti/ingest.py` — thin wrapper enforcing dream provenance stamps.
5. APScheduler job `dream_pass_{user_id}` registered via `@expose` in `backend/executor/scheduler.py` with `max_instances=1`, `CronTrigger` at user-local 3am, and a guard that bails if the user has been active in the last 15 minutes.

---

**Essential files for understanding this domain:**

- `autogpt_platform/backend/backend/copilot/graphiti/memory_model.py` — the MemoryEnvelope schema and all enum types
- `autogpt_platform/backend/backend/copilot/graphiti/ingest.py` — the per-user ingestion queue and worker; all write paths flow through here
- `autogpt_platform/backend/backend/copilot/graphiti/client.py` — group_id derivation, client cache, and the `get_graphiti_client` factory
- `autogpt_platform/backend/backend/copilot/graphiti/config.py` — all configurable knobs including embedder model selection and API key resolution
- `autogpt_platform/backend/backend/copilot/tools/graphiti_forget.py` — the only file that issues raw Cypher mutations; the soft-delete/hard-delete patterns to reuse
- `autogpt_platform/backend/backend/copilot/graphiti/context.py` — the warm-context read path showing how `retrieve_episodes` and `search` are combined
- `autogpt_platform/backend/backend/copilot/graphiti/_format.py` — attribute-resolution helpers for edge/episode objects across upstream versions
- `autogpt_platform/backend/backend/copilot/tools/graphiti_search.py` — scope filtering post-processing (`_filter_episodes_by_scope`) shows the full scope-handling pattern
- `autogpt_platform/backend/backend/executor/scheduler.py:596-620` — the `ensure_embeddings_coverage` and `optimize_block_descriptions` jobs are the closest existing conventions for `max_instances=1` background jobs to mirror
