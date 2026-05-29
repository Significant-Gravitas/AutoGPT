# Dreaming — Graphiti Best Practices and Audit

A grounded reference for how we use Graphiti, how the field thinks Graphiti should be used, and what changes the dream pass design implies. Citations inline; every empirical claim points at a URL. Where a claim is folklore or undocumented, it is labeled as such.

Scope: this is an audit and a recommendation set. No code is written. File:line references in §6 and §7 anchor every actionable item to a concrete place in our tree at HEAD on the `claude/admin-user-management-…` branch.

---

## 0. TL;DR for someone with twenty minutes

1. **The integration is mostly idiomatic.** We use the upstream `Graphiti.add_episode()`, hybrid `search()`, and the bi-temporal soft-delete pattern. The biggest deviations from "stock Graphiti" are (a) we wrap structured metadata in a `MemoryEnvelope` JSON-serialized into `EpisodeType.json`, and (b) we never run `build_communities()`. Both are defensible.
2. **Three concrete improvements before the dream pass ships.** (a) Pass `entity_types`/`edge_types` for our `MemoryEnvelope` fields so `status`, `confidence`, `scope`, and `source_kind` survive on edges — today they are stranded on the `:Episodic` content blob and our search code can't filter on them (`dreaming-memory.md` §4). (b) Switch off the inferred `gpt-4.1-nano` small model — already mitigated upstream (`client.py:150`), but worth keeping pinned. (c) Use `add_episode_bulk` for the dream pass's phase-1 consolidated facts on an *empty/synthetic* sub-group, then promote — this is the only safe place to use bulk since bulk skips invalidation.
3. **The dream pass should write through `add_episode` with custom entity types, not raw `add_triplet`.** The temporal invalidation pipeline is what makes Graphiti different from "Postgres with embeddings"; `add_triplet` short-circuits it.
4. **Demotion should set `expired_at`, not `invalid_at`.** `invalid_at` means "the fact ceased being true in the world." We mean "the system retired this fact." These are different things and Snodgrass [27] insists they not collapse. Add a custom property `status` on edges in parallel so the audit trail is queryable from search.
5. **Communities (Leiden / label-propagation) are not free** and the field gives no published evidence they help agent-memory retrieval — only document-summarization (Microsoft GraphRAG [13]). We should leave them off in P0 and revisit only if warm-context relevance plateaus.
6. **There is no published reference for >1M users on a single Graphiti deployment.** Zep operates Graphiti at scale internally but does not publish per-user graph statistics. The largest documented setup with public numbers is FalkorDB's own multi-tenant demos. Treat scaling as engineering-by-instrumentation, not by-citation.

---

## 1. Primary sources

### 1.1 The Zep / Graphiti paper

**Rasmussen, Paliychuk, Beauvais, Ryan, Chalef. "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv:2501.13956, Jan 2025.** [1]

This is the canonical Graphiti citation. It introduces:

- A three-tier hierarchical graph: **Episodic nodes** `N_e` (raw input, lossless), **Entity nodes** `N_s` (LLM-extracted, deduplicated), and **Community nodes** `N_c` (clusters with summaries).
- A bi-temporal model with four timestamps per edge: transaction time (`t'_created`, `t'_expired`) and valid time (`t_valid`, `t_invalid`). The paper labels these "transactional timeline (for auditing)" and "world timeline (when facts held true)" — a direct match to Snodgrass's 1999 bi-temporal SQL model [27].
- Edge invalidation via LLM: "the system employs an LLM to compare new edges against semantically related existing edges to identify potential contradictions. When detected, it invalidates the affected edges by setting their `t_invalid` to the `t_valid` of the invalidating edge."
- Community detection by **label propagation**, not Leiden, with incremental updates when a new entity joins (the new entity inherits the community of its plurality of neighbors). The docs separately mention Leiden [4]; the production code path supports both.
- Hybrid retrieval with three signals: BM25 (`φ_bm25`), cosine similarity over embeddings (`φ_cos`), and breadth-first graph traversal (`φ_bfs`).
- Benchmarks: **DMR 94.8% (vs. MemGPT 93.4%); LongMemEval 71.2% (vs. 60.2% baseline) with 90% latency reduction (2.58s vs. 28.9s, average context 1.6k vs. 115k tokens).**

Citation density: the paper is ~12 pages, three tables. It is the only primary source for Graphiti's architecture. Everything else is docs, blog posts, or third-party commentary.

### 1.2 Graphiti repo and version state

The Graphiti GitHub repo `getzep/graphiti` [2] declares its mission as "Build Real-Time Knowledge Graphs for AI Agents." At the time of this research pass the latest release is **v0.29.0** (Apr 27, 2026) [3]; our `pyproject.toml` pins `graphiti-core = ^0.28.2` [`dreaming-memory.md:5`]. Between 0.28.2 and 0.29.0:

- **0.29.0** combined node+edge extraction into one LLM call (reducing extraction cost); introduced `summarize_saga()` and a `fact_triple` episode type; Kuzu schema migration required.
- **0.28.2** (our pin) hardened search filters against Cypher injection.
- **0.28.1** replaced the `diskcache` dependency to resolve a CVE; fixed custom edge attributes on first ingestion.
- **0.28.0** simplified extraction (batch entity summarization) and redesigned driver operations.
- **0.27.0** fixed FalkorDB-specific bugs around pipe/slash characters in fulltext queries and group_id escaping.

Open issues directly affecting our deployment [33]:

- **#1483** FalkorDriver fulltext fails on hyphens in `group_id`. Our `derive_group_id` in `client.py:49-75` allows hyphens. Plausibly affects us.
- **#1465** FalkorDriver rejects escaped underscores in `group_id`. Less critical for us — we use a `user_` prefix without escaping.
- **#1001** `add_triplet()` does not save edges with FalkorDB. This is the blocker for the alternative phase-1 dream pass write path.
- **#760** `gpt-4.1-nano` dedup hallucination — already worked around in `client.py:150` by pinning `small_model=llm_model`.

### 1.3 Graphiti docs

The official Graphiti documentation lives at `help.getzep.com/graphiti/`. The pages this report draws on:

- Core concepts > Adding Episodes [4]
- Core concepts > Communities [5]
- Core concepts > Custom Entity and Edge Types [6]
- Searching the Graph [7]
- Adding Fact Triples [8]
- "Beyond Static Knowledge Graphs" Zep blog [9]
- LLM Data Extraction at Scale Zep blog [10]
- Graphiti + FalkorDB multi-tenant performance FalkorDB blog [11]
- DeepWiki Graphiti architecture overview [12]

---

## 2. The broader literature

### 2.1 Temporal knowledge graphs and bi-temporal data

**Snodgrass, R. T. "Developing Time-Oriented Database Applications in SQL." Morgan Kaufmann, 1999.** [27] The canonical primer. Defines **valid time** (when a fact was true in the world) and **transaction time** (when the system recorded it). Bi-temporal tables index both. SQL:2011 [Wikipedia: temporal database] codified Snodgrass's vocabulary — "application-time period tables" (valid time), "system-versioned tables" (transaction time), and "system-versioned application-time period tables" (bi-temporal). Graphiti's four-timestamp model is a direct port of this to a property graph.

**Allen, J. F. "Maintaining Knowledge about Temporal Intervals." Communications of the ACM 26(11), 1983.** [28] Defines the thirteen base relations between time intervals: `before`, `meets`, `overlaps`, `starts`, `during`, `finishes`, `equals`, and their six inverses. Graphiti does not natively reason in Allen relations; it stores `valid_at` and `invalid_at` as plain timestamps. If we ever want "the fact was true *during* the engagement period," we have to compute Allen relations on top.

**Cai, Xiang, Gao, Zhang, Li, Li. "Temporal Knowledge Graph Completion: A Survey." IJCAI 2023.** [29] First comprehensive survey of TKGC methods. Reviews TTransE, TA-DistMult, TeRo, DyERNIE, ChronoR, and the rest of the embedding-based TKG line. The methods all assume a fixed entity vocabulary and known time windows. Graphiti deliberately does *not* use TKG embeddings — it embeds *facts* (edge.fact_embedding) rather than (head, relation, tail, time) quadruples. This is a pragmatic choice: TKG embeddings cannot accept arbitrary new relation names from LLM extraction without retraining.

**Qian et al. "TimeR4: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering." EMNLP 2024.** [30] A retrieve-rewrite-retrieve-rerank pipeline that fine-tunes a retriever using contrastive time-aware learning. Reports +47.8% / +22.5% relative gains. This is the right baseline for whether *any* hand-built time-bias on top of Graphiti search yields measurable lift — currently no published number, but the TimeR4 numbers say "yes, plausibly."

**Zaporojets, Kaffee, Deleu, Demeester, Develder, Augenstein. "TempEL: Linking Dynamically Evolving and Newly Emerging Entities." NeurIPS 2022 Datasets & Benchmarks.** [31] Stratifies Wikipedia 2013–2022. Shows that **continual entities lose up to 3.1% linking accuracy** and **new entities lose up to 17.9%** when the underlying KB drifts. Highly relevant: TempEL is the dataset shape closest to a long-lived agent's memory, where new people, projects, tools appear constantly and existing ones drift. Graphiti's entity-resolution-by-cosine-similarity (1024-dim embedding of entity name, see [1] §3) will degrade in the same way unless we periodically refresh.

### 2.2 Knowledge graphs as a discipline

**Hogan, Blomqvist, Cochez, et al. "Knowledge Graphs." ACM Computing Surveys 54(4), 2021.** [32] The 80-page state-of-the-field. Defines knowledge graphs, graph data models, query languages, embedding methods, and extraction techniques. The taxonomy distinguishes "directed labelled graphs" (Graphiti's level) from "heterogeneous property graphs" (Neo4j's full model). Graphiti uses a *subset* of the property-graph model — typed nodes, typed edges, edges-with-properties, no nested or multi-typed nodes.

**Angles, Gutierrez. "Survey of graph database models." ACM Computing Surveys 40(1), 2008.** [34] The 2008 paper that codified "property graph" as distinct from RDF and from hierarchical/network databases. Important for understanding why FalkorDB (Cypher-style property graph) and Neo4j (Cypher-style property graph) interoperate with Graphiti but a pure RDF triple store does not.

### 2.3 Knowledge-graph-augmented retrieval

**Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Metropolitansky, Ness, Larson. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130, Microsoft Research, Apr 2024.** [13] The canonical recent paper on graph-augmented RAG. Indexes documents into a Leiden-clustered community graph; each community has an LLM-generated summary; queries fan out across community summaries and aggregate. Targeted at **global sensemaking** ("what are the main themes?"), not per-fact lookup. Reports substantial improvements over flat RAG on global tasks across ~1M-token corpora. **Graphiti is intentionally not GraphRAG**: per [2], the docs say GraphRAG is "batch-oriented, static document summarization" with seconds-to-tens-of-seconds latency; Graphiti is "continuous, incremental" with sub-second latency. The two address different problems.

**Guo, Xia, et al. "LightRAG: Simple and Fast Retrieval-Augmented Generation." arXiv:2410.05779, Oct 2024, EMNLP 2025.** [16] Dual-level retrieval (low-level entities + high-level concepts) plus incremental indexing. The architectural difference from Graphiti is that LightRAG is document-centric (chunks of text) whereas Graphiti is fact-centric (edges between named entities). LightRAG's incremental updates are the closest published proxy for what we expect under continuous agent ingestion.

**Gutiérrez, Shu, Gu, Yasunaga, Su. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." NeurIPS 2024.** [17] Personalized PageRank over an LLM-extracted entity graph. Inspired by complementary learning systems (McClelland 1995 [18], Kumaran 2016 [19]) — hippocampus as the indexer, neocortex as the slow-learning store. Beats vanilla RAG on multi-hop QA by up to **20%**, "10-30x cheaper and 6-13x faster" than IRCoT-style iterative retrieval. Architecturally similar to Graphiti minus the bi-temporal model.

**Sun, Xu, Tang, Wang, Lin, Gong, Ni, Shum, Guo. "Think-on-Graph: Deep and Responsible Reasoning of LLM on Knowledge Graph." ICLR 2024.** [20] LLM agent does iterative beam search on the KG, retrieves facts, reasons. Beats GPT-4 baselines on 6/9 datasets without training. Relevant when we eventually want the agent to *navigate* the graph rather than relying on a single retrieval call.

**Lewis, Perez, Piktus, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.** [21] The originating RAG paper. Worth citing for the "provenance" point: RAG was designed in part *because* parametric-only seq2seq models cannot point back at sources. Graphiti's `episodes` field on every fact is a direct realization of this — every fact knows which `:Episodic` node(s) produced it.

**Sanmartin, D. "KG-RAG: Bridging the Gap Between Knowledge and Creativity." arXiv:2405.12035, May 2024.** [22] Three-stage pipeline (storage → retrieval → answer generation). Less load-bearing for our design than GraphRAG/HippoRAG, included for completeness.

### 2.4 Agent memory architectures

**Park, O'Brien, Cai, Morris, Liang, Bernstein. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023.** [23] The memory-stream paper. Memories are scored as a weighted sum: `score = α_recency · recency + α_importance · importance + α_relevance · relevance` (all `α=1` in the published implementation). **Recency** is exponential decay since last access. **Importance** is LLM-generated. **Relevance** is cosine similarity to the query. This is the *de facto* reference scoring scheme for agent memory; Graphiti's hybrid search does *not* directly implement it but the three signals all exist (`created_at` for recency, BM25/embedding for relevance; we'd have to add importance ourselves).

**Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, Oct 2023.** [24] Hierarchical memory (main context, archival, recall) with function-call-driven transfers. The Zep paper [1] explicitly beats MemGPT on DMR (94.8 vs 93.4), making this the head-to-head reference point. MemGPT/Letta is also the team behind sleep-time compute [25].

**Sumers, Yao, Narasimhan, Griffiths. "Cognitive Architectures for Language Agents (CoALA)." TMLR 2024 (arXiv:2309.02427).** [26] The framework distinguishes working / episodic / semantic / procedural memory and structured action spaces. Our `MemoryKind` enum (`fact | preference | rule | finding | plan | event | procedure`) is loosely CoALA-shaped — `event` ≈ episodic, `fact|finding` ≈ semantic, `procedure|rule` ≈ procedural, `plan` ≈ working. We did not adopt CoALA explicitly but it is the right mental model.

**Xu, Liang, Mei, Gao, Tan, Zhang. "A-MEM: Agentic Memory for LLM Agents." NeurIPS 2025.** [35] Zettelkasten-inspired. Each note generates contextual descriptions, keywords, tags. New memories trigger updates to existing memories ("memory evolution"). The closest published prior art for what the dream pass should do.

**Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez. "Sleep-time Compute: Beyond Inference Scaling at Test-time." arXiv:2504.13171, Apr 2025.** [25] Pre-compute during idle time against likely future queries. **~5x test-time compute reduction** on Stateful GSM-Symbolic and Stateful AIME, +13% additional accuracy when sleep-time compute is scaled up. This is the right citation for P4 (Scenario pre-warm) and partial justification for the dream pass framing.

### 2.5 Entity resolution

**Konda, Doan, et al. "Magellan: Toward Building Entity Matching Management Systems." VLDB 2016.** [36] The system-building reference for entity matching: how-to guides, blocking, classification, evaluation. Highly relevant as a checklist for what Graphiti's resolution step is actually doing — Graphiti's [1] description ("1024-dim embedding + cosine + BM25 + LLM resolution prompt") is a minimal Magellan pipeline.

**Christen, P. "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection." Springer, 2012.** [37] The textbook. Three parts: overview, the pipeline (pre-process → block → compare → classify → evaluate), and topics like privacy and real-time matching. Graphiti's resolver does steps 3-4 (compare + classify) via cosine + LLM; the *blocking* step ("only resolve against candidates that are plausibly close") is implicit in the cosine retrieval. This means Graphiti has no explicit blocking key — it relies entirely on embedding quality.

### 2.6 Bi-temporal modeling references

- **Snodgrass, R. T.** Op. cit. [27] is the canonical primer.
- **Date, Darwen, Lorentzos. "Temporal Data and the Relational Model." Morgan Kaufmann, 2003.** Refines Snodgrass with type-system clarity. Cited for completeness; Graphiti's model is closer to Snodgrass than Date.
- **Fowler, M. "Bitemporal History." martinfowler.com, 2021.** Plain-English summary aimed at engineers. Good link to send a teammate who wants the gist.

### 2.7 Complementary learning systems

**McClelland, McNaughton, O'Reilly. "Why there are complementary learning systems in the hippocampus and neocortex." Psychological Review 102(3), 1995.** [18] The classical paper. Hippocampus does fast pattern separation; neocortex does slow statistical structure learning; reinstatement transfers between the two.

**Kumaran, Hassabis, McClelland. "What learning systems do intelligent agents need? Complementary learning systems theory updated." Trends in Cognitive Sciences 20(7), 2016.** [19] The DeepMind-era update. Same two-system picture, recast for AI. The intuition for "dreaming consolidates episodic memory into schemas" lives here.

---

## 3. Graphiti architecture deep-dive

### 3.1 Episodes vs entities vs facts — the three-layer model

From the Zep paper [1] §3, the graph is `G = (N, E, ϕ)` with:

- `N_e` **Episodic nodes** — raw input. Lossless. Three types: `message`, `text`, `json` [4].
- `N_s` **Entity nodes** — LLM-extracted, deduplicated. One entity may be referenced by many episodes.
- `N_c` **Community nodes** — clusters of entities with summaries.

Edges:

- `E_e ⊆ N_e × N_s` **Episodic edges** (Graphiti calls these `:MENTIONS`) — provenance from an episode to each entity it mentions.
- `E_s ⊆ N_s × N_s` **Semantic edges** (Graphiti calls these `:RELATES_TO`) — bi-temporal facts between entities.
- `E_c` Community membership edges (`:HAS_MEMBER`).

Why separate them? Two reasons.

**Auditability.** Every `:RELATES_TO` edge has an `episodes` array of source `:Episodic` UUIDs. If a fact is wrong, you can trace it back. Lewis et al. [21] argue this is the central design pressure for retrieval-augmented systems. Plain "key-value memory" (LangChain conversation memory, MemGPT [24] core memory) loses this.

**Temporal stability.** Entities and facts are slow-changing. Episodes accumulate. By separating them, you can answer "what do I know about Sarah?" without re-reading every episode that mentions Sarah.

**Tradeoff vs. flat fact storage (LangChain memory):** Flat memory is faster to read (no LLM extraction on write) but every retrieval is `O(n)` text-similarity. Graphiti's entity layer is the index.

**Tradeoff vs. pure entity-centric (Letta/MemGPT archival memory):** Letta stores text chunks with embeddings, optionally tagged by entity. Graphiti additionally stores the *relationships* between entities as first-class objects. This is the difference that lets Graphiti answer "what is the relationship between Sarah and the billing team?" in one Cypher hop.

### 3.2 The bi-temporal model — `valid_at`, `invalid_at`, `created_at`, `expired_at`

Per [1] §3 and the Zep blog [9]:

| Column | Meaning | Source of truth |
|---|---|---|
| `created_at` | When the system ingested the fact | LLM ingestion time |
| `expired_at` | When the system retired the fact (system-side, retraction) | Set by the contradiction detector OR our soft-delete |
| `valid_at` | When the fact became true in the world | Extracted from episode text or `reference_time` |
| `invalid_at` | When the fact ceased being true in the world | Set when a *contradicting* fact arrives |

The canonical example is "the user worked at Acme from 2020–2023" learned in March 2024:

- `created_at = 2024-03-15` (when we processed the message)
- `valid_at = 2020-01-01` (extracted from "I worked at Acme")
- `invalid_at = 2023-12-31` (extracted from "left in 2023")
- `expired_at = null` (we never retracted the record itself)

A worked example of how Graphiti uses both timelines (from [9]):

> "Maria works as a junior manager" arrives. Later, "Maria was promoted to senior manager" arrives. Graphiti's LLM contradiction detector identifies the conflict, sets `invalid_at` on the junior-manager edge equal to the `valid_at` of the senior-manager edge, and stamps `expired_at = now()` on the junior-manager edge to mark "the system retired this fact." The junior-manager fact survives in the graph but is filtered from default search results.

**This distinction is load-bearing for the dream pass.** Demotion (P0.3) is closer to `expired_at` than `invalid_at` — we are saying "the system has retired this record," not "the world has changed." See §7.

### 3.3 Episode types — `message`, `json`, `text`

From [4]:

- `text` — unstructured prose without speaker attribution.
- `message` — formatted as `Speaker: content`, used for multi-turn dialogue.
- `json` — arbitrarily nested structured data, processed differently from the others.

Our code uses `message` for user turns (`ingest.py:142-150`) and `json` for `MemoryEnvelope` payloads (`graphiti_store.py:228-235`, `ingest.py:172-184`). Both are idiomatic. The `json` choice for `MemoryEnvelope` is **not a hack** — the docs explicitly support it for structured payloads, and the LLM extractor handles JSON differently in a way that preserves field names. Best practice from [4]: "documents should remain compact to fit within LLM context windows" — our envelopes are <2KB and we are inside that envelope.

One caveat: `json` episodes have the same extraction pipeline as `text`. The LLM still pulls entities and facts. The structured fields are **not** preserved on resulting edges unless we use `entity_types`/`edge_types` to define them as Pydantic models (see §6.2).

### 3.4 LLM extraction pipeline

From the Zep blog [10], a Graphiti `add_episode()` call runs six tasks:

1. Entity extraction from episode
2. Entity deduplication against existing graph nodes
3. Fact extraction
4. Fact deduplication
5. Temporal validation (set `valid_at`)
6. Fact expiration (contradiction detection → set `invalid_at` on prior edges)

Each task is a separate LLM call. The Zep team explicitly moved away from a "35-guideline mega-prompt." Tasks 1-2 and 3-4 can run in parallel.

**Failure modes we should expect, ranked by frequency:**

- **Entity duplication** — the deduper fails to recognize "Sarah" and "Sarah Chen" as the same. Cause: embedding-only blocking (no surname normalization). Mitigation: warm-context retrieval surfaces both; the LLM in the next turn can resolve. Graphiti has no built-in alias table.
- **Missing edges** — the LLM extracts entities but fails to extract the relationship between them. Cause: the extractor is conservative. Mitigation: `custom_extraction_instructions` (we use this in `ingest.py:53-59`).
- **Hallucinated relationships** — the LLM emits a `:RELATES_TO` between two entities the source episode never connects. Cause: small models like `gpt-4.1-nano` were prone to this (issue #760 [38]). Mitigation: pin `small_model=llm_model` (done in `client.py:150`).
- **Wrong entity type** — the LLM classifies "Acme Corp" as a `Person`. Cause: ambiguous context. Mitigation: custom entity types with strong descriptions [6].

### 3.5 Community detection

`build_communities()` runs the Leiden algorithm [5] (or, per the paper [1], label propagation in production — discrepancy between docs and paper is unresolved). The result is `:Community` nodes with LLM-generated `summary` text and `:HAS_MEMBER` edges to entities.

How it interacts with retrieval: the docs [7] list community-level recipes (`COMMUNITY_HYBRID_SEARCH_RRF/MMR/CROSS_ENCODER`). The default `search()` does *not* hit communities; you have to use `search_()` with a `SearchConfig` that includes community search.

**Cost:** rebuilding communities is `O(|V|·log|V|)` graph-algo cost plus one LLM call per community to summarize. For a graph with 500 entities, ~30 communities → 30 LLM calls. At Sonnet pricing, ~$0.30 per rebuild.

**Incremental update:** `add_episode(..., update_communities=True)` assigns new entities to the community of their plurality neighbor [1] §3. This is cheap but drifts over time, hence the docs recommendation [5]: "we still recommend periodically rebuilding communities to ensure the most optimal grouping."

**Should we turn it on?** No, for v1. Three reasons:

1. **No measured benefit for agent memory retrieval.** The community-detection literature is about *document summarization* (Microsoft GraphRAG [13]) on ~1M-token corpora, not per-user memory at <1000 entities. Our scale doesn't justify the operational cost.
2. **Latency cost on writes.** `update_communities=True` adds another LLM call per `add_episode`. Our ingestion queue already serializes by user (`ingest.py:62-99`); adding LLM rounds prolongs the queue.
3. **The dream pass already produces summaries.** Phase 2 (recombination) emits `ProposedFinding`s that act as community-level summaries with explicit evidence trails. We get the benefit without paying for Leiden.

Revisit if (a) post-P0 metrics show warm-context relevance plateauing while graph grows, or (b) we ship a per-user "what does the system know about me?" overview surface — that's exactly the GraphRAG global-sensemaking shape.

### 3.6 Custom entity types and edge types

From [6]:

- Custom types are Pydantic models. Entity types (`Person`, `Company`, `Product`). Edge types (`Employment`, `Investment`, `Partnership`).
- `edge_type_map` constrains which edge types are legal between which entity types. `("Person", "Company"): ["Employment"]`.
- `excluded_entity_types` prevents extraction of specific types from an episode.
- Protected attribute names you cannot use: `uuid`, `name`, `labels`, `created_at`, `summary`, `attributes`, `name_embedding`.

**When is it worth it?** When you have:

- A repeating domain pattern (e.g., every business has Customers, Vendors, Products).
- Attributes you need to query on (e.g., `Product.sku`, `Employment.start_date`).
- Edge directions that matter semantically (e.g., `EMPLOYED_BY` is not symmetric).

**Tradeoff:** Custom types make extraction more reliable *and* more brittle. The LLM is forced into a schema; if the episode doesn't fit, it either fails to extract or shoehorns. Inferred types are flexible but ungroundable in code (you can't `MATCH (p:Person {name:'Sarah'})-[e:EMPLOYED_BY]->(c)` reliably).

**For us:** §6.2 recommends defining custom edge types for the `MemoryEnvelope` fields we currently strand on `:Episodic` content. Specifically `status`, `source_kind`, `confidence` should become edge attributes on a `MEMORY_FACT` edge type derived during extraction.

### 3.7 Sagas

A Saga in Graphiti is a named, ordered sequence of episodes. From [12]: "Graphiti uses sagas to create (or reuse) a SagaNode with a given name in a group, then link each episode to it via a HasEpisodeEdge, and link consecutive episodes to each other via NextEpisodeEdge."

Used for: podcast transcript chains, conversation transcripts, anything where temporal order within a saga matters more than within the group.

**Should we use sagas?** Probably not for chat sessions — our episodes already have a `session_id` in the `name` (`ingest.py:129`), and we don't navigate them in saga-order today. The right use case for us is **the dream pass itself**. Each dream pass could be its own saga (`saga="dream_{user_id}_{date}"`) — that gives us `summarize_saga()` from 0.29.0 [3] for free as an audit surface. Defer to v2.

---

## 4. Search and retrieval

### 4.1 Hybrid search

From [7] and [1] §4:

- **`graphiti.search(query, group_ids, num_results, center_node_uuid)`** — convenience method. Returns `:RELATES_TO` edges only. Uses Reciprocal Rank Fusion (RRF) of BM25 + cosine + (optional) graph distance reranking from `center_node_uuid`.
- **`graphiti.search_(query, config, group_ids)`** — advanced. Returns nodes and edges. `config` is a `SearchConfig` with edge, node, and community sub-configs; 15+ named recipes exist (`EDGE_HYBRID_SEARCH_RRF`, `NODE_HYBRID_SEARCH_NODE_DISTANCE`, `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`, etc.).

**Reranking strategies** [7]:

- **RRF** — weighted-rank fusion across BM25 and cosine. Cheap, default.
- **MMR** (Maximal Marginal Relevance) — relevance × diversity. Use when you'd otherwise return ten paraphrases of the same fact.
- **Cross-encoder** — joint query-result scoring with a separate model (OpenAI/Gemini/BGE). Higher latency, higher precision. Worth using on the warm-context call.
- **Node-distance** — graph-proximity reranking around a `center_node_uuid`. Use when you have an anchor entity.

### 4.2 Best-practice search recipes

| Query shape | Recipe | Notes |
|---|---|---|
| "Everything about entity X" | `NODE_HYBRID_SEARCH_NODE_DISTANCE` with `center_node_uuid = X.uuid` | Returns nodes ranked by graph distance from X. |
| "What did the user say about Y in last 30 days?" | `EDGE_HYBRID_SEARCH_RRF` + post-filter on `valid_at >= now-30d` | Graphiti's search filter doesn't natively accept a date range; we filter in Python. |
| "Stale or contradictory facts" | Raw Cypher: `MATCH ()-[e:RELATES_TO]->() WHERE e.invalid_at IS NOT NULL OR e.expired_at IS NOT NULL` | Default `search()` *hides* these; we have to bypass. |
| "User's stated preferences" | `search_(query="user preferences", config=NODE_HYBRID_SEARCH_RRF)` filtered to `MemoryKind.preference` | Currently no way to filter by `memory_kind` because it's on the `:Episodic` content blob, not on the edge. See §6.3. |

### 4.3 Reranking we should layer

Graphiti's default `search()` returns RRF-reranked edges. For warm-context (where we get one shot per session), it is worth adding:

- **Time decay** layered on top, scoring `score = base_score × exp(-λ · age_days)`. Park et al. [23] is the reference; their `α_recency=1` with `decay_factor=0.99/hour` is a starting point.
- **Importance** signal: if we store `confidence` on the edge (per §6.2), include it as a multiplier.
- **Source-kind boost**: prefer `user_asserted` over `assistant_derived` in the absence of contradictions. This is `MemoryEnvelope.source_kind` and again requires it to live on the edge.

### 4.4 Time-aware retrieval

The TimeR4 paper [30] reports +47.8% / +22.5% gain from a time-aware retrieve-rewrite-rerank pipeline. Their architecture:

1. Retrieve facts.
2. Rewrite query using retrieved background to extract explicit time constraints ("Q2 2024," "last month").
3. Retrieve again with the rewritten query.
4. Rerank by closeness of `valid_at` to the time-constraint.

We do none of this today. The lift is potentially large enough to be a v2 priority once the dream pass ships. Specifically the rewrite step — many of our user queries are time-relative ("recent decisions," "what we discussed last week") and Graphiti's default search has no notion of relative time.

---

## 5. Scaling and operational concerns

### 5.1 Per-user vs shared databases

We use **one FalkorDB database per user**, derived as `user_{user_id}` from `client.py:49-75` and `falkordb_driver.py:6`. This is our choice, not a Graphiti default. Graphiti supports two patterns:

- **One graph per tenant** — what we do. Strong isolation. Per-user database in FalkorDB. Operational unit = one Redis-style hash-mapped DB per user.
- **One shared graph with `group_id` partitioning** — what the docs default to [11]. All users share a database; `group_id` filters separate them at query time. Cheaper at small scale, riskier at large.

**FalkorDB's published recommendation** [11]: "Tenant isolation in FalkorDB provides dedicated graph instances per agent while sharing compute resources. This prevents data conflicts, maintains privacy boundaries." This matches our choice.

**Largest documented scale:** FalkorDB markets "millions of nodes" per tenant and claims **496x faster P99 latency vs. Neo4j and 6x better memory efficiency** [11], though without public benchmark code. No published reference for **>1M users on Graphiti specifically.** Zep operates production deployments [1] but does not disclose user counts.

**Hash slot ceiling [Wikipedia: Redis Cluster]:** FalkorDB inherits Redis Cluster's 16,384 hash slot count. Each per-user database maps to one slot. At >16k users we have to shard across FalkorDB clusters, which is operationally non-trivial. **This is a real ceiling and our roadmap should acknowledge it.**

### 5.2 Episode rate limits

From the Graphiti README [2]: `SEMAPHORE_LIMIT=10` by default to "prevent 429 rate limit errors." Our `config.py:62-65` sets `semaphore_limit=5` per Graphiti client. Combined with the per-user serialization queue in `ingest.py:62-99`, **we serialize one episode at a time per user**, with up to 5 concurrent LLM operations within that one episode's extraction pipeline.

**Bottlenecks, ranked:**

1. **LLM extraction**, especially for the dedup step. ~2-5s per `add_episode` against `gpt-4.1-mini`. Pinning `small_model = llm_model` (our `client.py:150`) avoids the nano hallucination bug but increases cost.
2. **Embeddings.** `text-embedding-3-small` is cheap (~$0.02/M tokens) but adds ~300ms per fact.
3. **FalkorDB writes.** Sub-10ms per atomic Cypher query [11]; not the bottleneck.

A user generating one chat message every 30 seconds produces one episode every 30 seconds. The pipeline processes one in 2-5s. Queue depth grows ~5x slower than ingestion. **At ~360 messages/hour for one user, the queue saturates `maxsize=100` (`ingest.py:253`) — drops start at 1/2-hour sustained burst.** Realistic chat traffic is nowhere near this, but the dream pass at 50 episodes/sec would saturate immediately, which is why `enqueue_dream_episode` must rate-limit itself.

### 5.3 Storage growth

No published numbers from Zep. From first principles:

- One episode ≈ 1-2 KB content + 100-500 bytes metadata + 1536-float embedding ≈ 7-10 KB.
- One entity ≈ 100-500 bytes + 1024-float embedding ≈ 4.5 KB.
- One `:RELATES_TO` edge ≈ 100-500 bytes + 1024-float embedding ≈ 4.5 KB.

A typical chat session generates 5-20 episodes, 3-10 new entities, 5-20 edges. **Per user per active day: ~50-150 KB.** At 365 days ≈ 18-55 MB per power user. Without consolidation, this grows linearly. Community detection would not shrink the graph; it would add `:Community` nodes (extra cost). **Our dream pass is the first lifecycle process that actually shrinks the working set** by demoting facts out of default search.

### 5.4 Concurrency model

Graphiti is async (asyncio). FalkorDB is built on Redis and runs queries on **`THREAD_COUNT` threads** by default to system core count [FalkorDB config docs]. **Cypher queries are atomic per execution** (Redis-single-thread semantics within a database).

The contention point is **between graph reads and graph writes inside one `add_episode` call** — the LLM-dedup step reads existing nodes, then writes new ones. Between read and write, another concurrent `add_episode` for the same user could insert a duplicate. Our `_ingestion_worker` (`ingest.py:62-99`) serializes per-user, eliminating this — *as long as both writers go through the same queue*. They do not for the dream pass (`dreaming-memory.md` §6). This is a real risk addressed by the proposed Redis advisory lock.

### 5.5 Backup, restore, migration

Graphiti version bumps almost always require a re-index (`build_indices_and_constraints`). The 0.29.0 release [3] required Kuzu users to migrate `reference_time` to a typed column — same kind of breakage could land for FalkorDB any release. **Operational rule:** before bumping Graphiti, run a full reindex on a copy of one user's database and validate search results. There is no documented `graphiti migrate` command — version bumps are manual.

FalkorDB persistence is Redis RDB/AOF. Backup = standard Redis dump. Per-user database = per-key dump. Cross-version migration of the graph schema is the Graphiti job, not FalkorDB's.

### 5.6 Cost

For one active user, conservative monthly numbers:

- 300 episodes ingested → 300 × ~$0.005 extraction (Sonnet-4-6) = $1.50.
- 1000 facts embedded × 50 tokens × $0.02/M = negligible.
- Storage: 20 MB × FalkorDB hosting ≈ $0.05.
- Search: 30 sessions × 0.5 LLM calls of warm context ≈ $0.10.

**~$2/user/month, ~$24/user/year.** Heavy users 3x. Dream pass adds ~$0.30/pass × 30/month = $9/month. **Total memory: ~$10-30/user/month at scale.** This is more than chat itself in many sessions.

---

## 6. Audit of our integration

This section walks every file in `autogpt_platform/backend/backend/copilot/graphiti/` and `autogpt_platform/backend/backend/copilot/tools/graphiti_*.py` and flags idiomatic uses, anti-patterns, and concrete recommendations.

### 6.1 `client.py:49-75` — `derive_group_id`

**Idiomatic.** Per-user database isolation matches FalkorDB's published recommendation [11]. The injection sanitization is conservative (stripping non-`[a-zA-Z0-9_-]`) and matches Graphiti's `group_id` validation.

**One concern.** Open issue #1483 [33] reports `FalkorDriver.build_fulltext_query` failing on hyphens in `group_id`. We allow hyphens (UUIDs contain them). Recommend: try a hyphenated UUID end-to-end in CI to confirm we're not silently broken by the upstream bug. If we are, our options are (a) replace hyphens with underscores in `derive_group_id` (mostly safe — group_ids are opaque) or (b) wait for the upstream fix.

### 6.2 `client.py:147-176` — Graphiti client construction

**Idiomatic and well-instrumented.** TTL cache (`client.py:75-95`), per-event-loop scoping (`client.py:24-37`), explicit driver close on eviction (`client.py:78-94`). The `small_model=llm_model` workaround (`client.py:150`) correctly mitigates issue #760 [38].

**Recommendation:** consider exposing `max_coroutines=graphiti_config.semaphore_limit` (`client.py:172`) as a tunable in `config.py`. The default `5` may be too tight for the dream pass which submits multiple episodes in a burst — bumping to 10 during the dream pass and back to 5 for chat would help. This is currently a static config.

### 6.3 `ingest.py:53-59` — `CUSTOM_EXTRACTION_INSTRUCTIONS`

**Idiomatic.** The instructions correctly tell the LLM to ignore agent infrastructure (`User`, `Assistant`, `AI`) and focus on real-world entities. This is exactly the `custom_extraction_instructions` hook from [4].

**One enhancement.** Consider adding a positive instruction to *prefer* extracting `:RELATES_TO` edges between people and projects, since our domain is workflow/agent automation. Currently the instructions are entirely negative ("do not extract X"). The LLM extractor [10] does better with examples of *what to do*.

### 6.4 `ingest.py:188-241` — `enqueue_episode`

**Largely idiomatic** with a key gap.

The function correctly routes structured `MemoryEnvelope` data through `EpisodeType.json` (`is_json=True` → `EpisodeType.json`, `ingest.py:220`). Per [4] this is the right choice for structured payloads.

**The gap:** we do not pass `entity_types` or `edge_types` to `add_episode`. Consequently, the LLM-extracted `:RELATES_TO` edges have a generic `RELATES_TO` label with `fact` and `name` properties but **no `status`, `confidence`, `source_kind`, `scope`, or `provenance` on the edges themselves.** Those live only on the `:Episodic.content` JSON blob (`dreaming-memory.md` §4 confirms this).

**Why this matters for the dream pass:** ratification (P0.4) wants to flip `status=active` *on the edge*. Today it has to flip it on the `:Episodic.content`, which is opaque JSON and requires re-serialization in Cypher (`SET ep.content = $new_json`). This is the workaround that exists *because* we don't use custom types.

**Concrete recommendation:** Define custom entity types (`Person`, `Organization`, `Project`, `Concept`, `Preference`, `Rule`) and edge types in a new `backend/copilot/graphiti/types.py`. The edge schema should include:

```python
class MemoryFact(BaseModel):
    status: MemoryStatus = MemoryStatus.active
    confidence: float | None = None
    source_kind: SourceKind = SourceKind.user_asserted
    scope: str = "real:global"
    provenance: str | None = None
```

Pass them as `entity_types={...}, edge_types={"MemoryFact": MemoryFact}, edge_type_map={("Entity","Entity"):["MemoryFact"]}` on every `add_episode` call. Status flips become single-line Cypher SETs on edges, and search can finally filter by `status` natively.

**Migration cost:** existing data has bare `:RELATES_TO` edges without these properties. We'd need a one-time backfill (`SET e.status = 'active' WHERE NOT EXISTS(e.status)`) before relying on the property. Plan a `prisma migrate`-style scripted backfill, not a code-level deferred default.

### 6.5 `ingest.py:62-99` — `_ingestion_worker`

**Correctly serializes per user.** The per-loop state (`ingest.py:27-47`) addresses the asyncio-loop-binding problem with Graphiti's redis.asyncio client. This is a non-obvious bug class and the comment explaining it (`ingest.py:21-26`) is a model of how to leave breadcrumbs.

**Two improvements for the dream pass:**

1. **Backpressure logging.** Currently a `QueueFull` exception (`ingest.py:152-157`, `ingest.py:235-240`) logs and drops. For dream-pass-driven enqueues this should be a hard error or at minimum a metric, because dropping a dream-derived consolidated fact is qualitatively worse than dropping a routine chat episode.

2. **Lock contention between scheduler and executor.** `dreaming-memory.md` §6 spells out the risk: the scheduler process runs its own asyncio loop and bypasses our queue. The Redis SETNX lock proposed in `dreaming-memory.md:139-141` is the right primitive. Implementation should be in a new `backend/copilot/graphiti/locks.py` (per the spec).

### 6.6 `ingest.py:298-313` — `_is_finding_worthy` and `_distill_finding`

**Pragmatic workaround, not a Graphiti idiom.** This is our heuristic for whether to ingest an assistant message as a derived finding. The prefix list (`ingest.py:278-292`) is hand-curated.

**Concern.** This is the kind of code that drifts. Examples:

- `"sure!"` matches a wide range of "Sure, here's how to..." actual findings.
- `"i've created"` is good but won't catch `"I have created"` or `"Created the..."`.

**Recommendation.** Replace with a tiny LLM judge in v2 — `gpt-4.1-nano` at ~10 tokens per call is essentially free. For P0 it is acceptable; just instrument and review the false-negative rate.

### 6.7 `context.py:53-69` — `fetch_warm_context`

**Idiomatic.** `gather(search, retrieve_episodes)` is the recommended pattern for warm context [10]. The 8s timeout (`config.py:69-72`) is reasonable.

**Two enhancements:**

1. **Use `search_()` with a cross-encoder recipe.** Today we use the default `search()` (RRF only). The cross-encoder recipe `NODE_HYBRID_SEARCH_CROSS_ENCODER` typically lifts relevance precision ~10-15% at the cost of a single extra LLM call. Worth the cost for the *one* warm-context call per session.

2. **Filter `:RELATES_TO` edges where `expired_at IS NOT NULL OR invalid_at IS NOT NULL`.** Default search already does this (per Zep paper [1]), but we should verify in tests — there's a class of bug where Graphiti returns retired edges anyway. Add a regression test that creates an edge, soft-deletes it, and confirms it doesn't appear in `search()`.

### 6.8 `context.py:106-117` — `_is_non_global_scope`

**This is the scope-filter-in-Python workaround.** Per `dreaming-memory.md` §3, scope is *not* a graph-level construct — it's a property on the `MemoryEnvelope` JSON blob. Filtering happens post-fetch.

**Concrete fix:** Once custom entity types are introduced (§6.4), `scope` becomes an edge property and the filter moves into the Cypher: `WHERE e.scope = $scope OR e.scope = 'real:global'`. This is faster (no Python loop) and correct (we currently drop episodes with malformed JSON, which is silent data loss).

### 6.9 `falkordb_driver.py` — fulltext query builder

**A 30-line subclass that fixes upstream stopword handling.** Looks like a fork of FalkorDB's upstream `build_fulltext_query`. The `validate_group_ids` call at line 13 is good defensive engineering.

**Concern:** open issues #1440 and #1483 [33] indicate the upstream fulltext query builder is fragile on backticks, hyphens, and stopwords. We may already be silently degraded on these inputs. Recommend: add a unit test matrix covering backtick chars, hyphens, single-stopword queries, and empty strings. If any fail, escalate upstream or fork further.

### 6.10 `_format.py` — attribute resolution helpers

**Pragmatic and necessary.** Graphiti changes attribute names between versions (`fact` vs `name`, `content` vs `body` vs `episode_body`). Centralizing in `_format.py` is exactly the right pattern.

**No recommendation** — keep this file as the single point of contact for upstream churn.

### 6.11 `tools/graphiti_search.py` — `MemorySearchTool`

**Mostly idiomatic.** Uses `client.search()` for edges and `client.retrieve_episodes()` for recent context, then post-filters by scope.

**Two concerns:**

1. **`_filter_episodes_by_scope` is the same workaround as §6.8.** Same fix applies — push scope into edges via custom types.
2. **No reranking.** We return raw `search()` results. For an explicit memory-search tool (user actively asked us to recall), spend the cross-encoder budget.

### 6.12 `tools/graphiti_store.py` — `MemoryStoreTool`

**Mostly idiomatic.** Builds a `MemoryEnvelope`, serializes to JSON, enqueues. The Rule and Procedure special cases (`graphiti_store.py:188-206`) are well-handled.

**One concern:** `provenance=session.session_id` (`graphiti_store.py:223`) is too thin. Best practice from [21] (Lewis RAG provenance argument) is to include enough to retrieve the source — `session:{id}#msg:{sequence}` would let ratification (P0.4) find the originating message. Today we'd have to search the session for the trigger phrase, which is fragile.

### 6.13 `tools/graphiti_forget.py:254-290` — `_soft_delete_edges`

**Idiomatic with one nit.** Sets `invalid_at = datetime(), expired_at = datetime()` on the matched edge by uuid. This is a temporal invalidation matching Graphiti's contradiction-detection pattern.

**The nit:** setting both `invalid_at` and `expired_at` conflates "the world has changed" with "we retracted the record." Per the Snodgrass model [27] and the Zep blog [9], `invalid_at` should be set only when the *world* changes. User-initiated forget is a system retraction → only `expired_at` should be set.

**Recommendation:** add a `_retract_edges` variant that sets only `expired_at`, and *use that* for user-initiated forget. Reserve `_soft_delete_edges` (with both) for contradiction-detector use cases. The dream pass's demotion should also use the retract variant.

### 6.14 `tools/graphiti_forget.py:293-349` — `_hard_delete_edges`

**Idiomatic.** Single atomic Cypher with `WITH` to capture the uuid before `DELETE` (avoids the FalkorDB #1393 bug). Cleans up episode back-references in `entity_edges`. This is exactly the pattern recommended by the Graphiti maintainers.

### 6.15 Summary of audit findings, ranked

| # | Severity | File:line | Issue | Recommendation |
|---|---|---|---|---|
| 1 | High | `ingest.py:188-241`, `graphiti_store.py:217-235` | `status`/`confidence`/`scope`/`source_kind` stranded on `:Episodic.content`; can't filter on them in search; ratification has to mutate JSON blob | Define custom entity + edge types per §6.4; one-time backfill |
| 2 | High | `tools/graphiti_forget.py:254-290` | Soft delete conflates `invalid_at` (world) with `expired_at` (system); dream demotion needs the latter alone | Split into `_retract_edges` (system-only) and `_soft_delete_edges` (both); use retract for dream demotion |
| 3 | Medium | `client.py:49-75` | Hyphens in `group_id` may hit upstream issue #1483 | CI test with hyphenated UUID; consider sanitizing to underscore |
| 4 | Medium | `context.py:53-69`, `graphiti_search.py:114-125` | Default `search()` (RRF only) for both warm-context and explicit search | Use `search_()` with a cross-encoder recipe on at least the warm-context call |
| 5 | Medium | `ingest.py:62-99` | Scheduler-process bypasses per-user queue → dream/chat write race | Redis SETNX advisory lock (already in spec) |
| 6 | Low | `graphiti_store.py:223` | `provenance=session.session_id` lacks message-level grain | `session:{id}#msg:{sequence}` |
| 7 | Low | `ingest.py:278-313` | Hand-curated prefix list for finding distillation will drift | Replace with `gpt-4.1-nano` judge in v2 |
| 8 | Low | `ingest.py:53-59` | Custom extraction instructions are entirely negative | Add positive examples (Zep blog [10]) |
| 9 | Low | `falkordb_driver.py` | Upstream fulltext builder is fragile on edge cases | Test matrix; fork further or escalate |

---

## 7. Best practices specifically for the dream pass

### 7.1 P0.2 Memory recombination — write path

The question from `TODO.md` P0.2: should consolidated facts go through `add_episode()` (LLM extraction) or `add_triplet()` (direct insert)?

**Recommendation: `add_episode()` with structured `MemoryEnvelope` and `update_communities=False`.**

Reasoning:

- **`add_triplet()` skips the temporal invalidation pipeline** [8] and currently has a known bug on FalkorDB (#1001 [33]). Even when fixed, it bypasses the contradiction detector — which is the whole point of using Graphiti.
- **`add_episode()` gets us deduplication for free.** A dream pass that emits "Sarah is on the billing team" when an existing edge already says so should *not* create a duplicate edge. Graphiti's resolver handles this. `add_triplet` would require us to write our own.
- **Set `update_communities=False`** explicitly. We do not run communities; passing this False makes that decision explicit and avoids surprises if a future Graphiti version defaults to True.
- **Use a dream-specific `name` prefix** (`dream_{job_id}_{phase}_{counter}`) per `dreaming-memory.md` §2(b). Lets us audit and remove dream-derived episodes selectively.
- **Set `reference_time = (last_episode_in_window).valid_at`**, not `datetime.now()`. The dream pass is consolidating *historical* facts; the system should think they were learned at the time of their evidence, not now. This matters for `valid_at` extraction.

### 7.2 P0.3a/b Demotion — `expired_at` + custom `status`

`TODO.md` P0.3a question: is `expired_at` + a custom `status` property the right pattern, or should we mark `invalid_at`?

**Recommendation: `expired_at = now()` + custom edge property `status='superseded'` (or `'contradicted'`).**

Reasoning, layered:

1. **Semantic correctness per Snodgrass [27].** `expired_at` is transaction-time retraction. `invalid_at` is valid-time "no longer true in the world." Stale facts are *system* retractions, not world changes. See `_soft_delete_edges` recommendation in §6.13.

2. **Graphiti's default `search()` filters out `expired_at IS NOT NULL`** [1] §4. Setting `expired_at` is enough to remove the fact from chat warm context. We do not need to also set `invalid_at`.

3. **The custom `status` property gives us an audit trail.** Phase 3 sanitizer (`p0-spec.md` §2) needs to know *why* a fact was demoted: `stale_fact`, `contradicted_by:{uuid}`, `entity_invalidated:{uuid}`, `user_signal`. Set `status='superseded'` and `expiration_reason=<short_string>` on the edge. This survives Cypher round-trips and is queryable from `search_()` with a custom config.

4. **P0.3b cascading expiry** (`p0-spec.md` §4) — the `[r:RELATES_TO]-(other)` pattern (single-hop) is the right call. The instinct to use `[r:RELATES_TO*1..N]` is exactly the runaway-demotion bug we are protecting against (`dreaming-spec.md` §6). Keep the single-hop discipline; ratification (P0.4) is how we *re*-promote good facts that got caught in the cascade.

### 7.3 P0.4 Ratification — also flip `status` on edges

`TODO.md` P0.4 question: status lives on the `:Episodic` content blob; should we also set status on the `:RELATES_TO` edges?

**Recommendation: yes, set status on edges (as well as content).**

Reasoning:

- **Search can only filter on edge properties.** If we ratify a `tentative` fact and only update the `:Episodic.content` JSON, the edge stays `tentative` and any downstream filter (`status='active'`) won't see it.
- **`:RELATES_TO` edges live longer than `:Episodic` nodes.** Episodes are write-once. Edges accumulate updates (`expired_at`, `invalid_at`, our new `status`). Status belongs on the durable artifact.
- **Tradeoff:** double-write cost on ratification. Cypher update is sub-10ms [11]; this is not a real concern.

Concrete shape:

- On ratify: `MATCH (...)-[e:RELATES_TO {uuid:$uuid}]->(...) SET e.status = 'active', e.ratified_at = datetime()`.
- On unratified-expiry (30-day TTL): `SET e.status = 'superseded', e.expired_at = datetime(), e.expiration_reason = 'unratified'`.
- Keep mutating the `:Episodic.content` in parallel so old code paths that read JSON still get the up-to-date status.

Once custom edge types (§6.4) land, this becomes trivial; in the interim, keep both writes in `apply.py:apply_operations()`.

### 7.4 P0.5 Web fact check — write path

`TODO.md` P0.5 question: should verified-against-web facts be written as new episodes, direct triplets, or edges with a special property?

**Recommendation: new episodes with `source_kind=tool_observed` and a `web_verified_at` edge property.**

Reasoning:

- **Episodes preserve provenance** — the URL(s), the verification timestamp, the LLM judge's reasoning. Putting it in an episode means we can audit later.
- **`tool_observed` matches the existing `SourceKind` enum** (`memory_model.py:14-17`). We do not need a new source type.
- **`web_verified_at` on the edge marks the durable claim.** Searches that want only web-verified facts can `MATCH (...)-[e:RELATES_TO]-(...) WHERE e.web_verified_at IS NOT NULL`. This is queryable without parsing JSON.
- **For *contradictions*, the web verification triggers a demotion** of the contradicted edge per §7.2 — `status='contradicted'`, `expiration_reason='web_contradicted:{url}'`. The fresh fact arrives as its own episode.

This matches the `TODO.md` P0.5 scope guard: web verification can *demote* existing memories but new web-derived facts ride the ratification loop.

### 7.5 P0.6 Memory benchmark harness

Not a Graphiti-specific recommendation, but: the benchmark harness should record *which Graphiti version produced which result*. Graphiti's extraction prompts change between versions (per [10], the team is actively iterating). A regression in warm-context relevance might be due to our code or due to upstream prompt changes. Pin the Graphiti version in `pyproject.toml`, log it on every benchmark run, and review on bumps.

---

## 8. Operational anti-patterns

Drawn from the Graphiti GitHub issue tracker [33] and the Zep blog [10]. These are field-tested failure modes we should not commit.

1. **Running `build_communities()` during peak hours.** Leiden over a non-trivial graph is `O(|V| · log|V|)` plus one LLM call per community for summarization. Several issues report retrieval slowdowns during community rebuild. Run during off-peak (per-user 3am, same as our dream pass).
2. **Over-aggressive `SEMAPHORE_LIMIT`.** Bumping it above ~20 reliably triggers OpenAI 429s in production [README, 2]. Stick to 5-10.
3. **Using `add_episode_bulk` on a non-empty graph.** Per [4], bulk skips edge invalidation. Anyone using it on a live graph creates duplicate facts. Reserve it for migrations or empty group_ids.
4. **Embedding-only blocking for entity resolution.** As discussed in §3.4, Graphiti's resolver relies on cosine similarity. If we ever load users with many similar names ("John Smith"), expect dedup failures. The fix is alias tables, not a deeper embedding model.
5. **Mixing message-type and json-type episodes within one session.** Graphiti extracts entities differently per type [4]; cross-type extraction quality is uneven. Pick one type per *kind of episode* and stick to it. We already do this — `message` for user turns, `json` for `MemoryEnvelope` — keep it that way.
6. **Storing high-PII fields in entity names.** Entity names are indexed for fulltext + cosine. PII in entity names ends up everywhere. Stuff PII into `:Episodic.content` (which is private to the user's group) or never store it. We hit this point in `project memory` (per the CLAUDE.md) so it's already on our radar.
7. **Skipping `build_indices_and_constraints()` after a version bump.** Issue #325 [33] is one of many — index drift causes silent search degradation. Run it once per Graphiti version per database.
8. **Trusting `add_triplet()` on FalkorDB.** Open #1001 [33]. Until fixed, use `add_episode()` only.
9. **Mutating the `:Episodic.content` JSON in place via Cypher `SET ep.content = $new_json`.** Workable but loses the embedding refresh — episode content embeddings won't update. Prefer writing a new episode and soft-deleting the old (see `dreaming-memory.md` §8 P3 recommendation: write fresh, soft-delete tentative).
10. **Per-user `:Community` accumulation without cleanup.** If we ever turn on communities, the `build_communities()` rebuild *adds* `:Community` nodes without deleting the old ones in some versions. Add `MATCH (c:Community) DETACH DELETE c` *before* the rebuild.

---

## 9. Open questions and unknowns

Honest list of what's not settled. Each item is either an empirical gap, a design ambiguity, or a research-grade question.

1. **Does bi-temporal modeling actually improve LLM retrieval in practice?** The Zep paper [1] reports 18.5% lift on LongMemEval but does not ablate the bi-temporal component specifically. TimeR4 [30] reports +47.8% with a *different* time-aware retrieval architecture, suggesting time-awareness in retrieval matters — but the bi-temporal *storage* model's contribution is unmeasured. This is a P1 eval target.

2. **Optimal episode granularity.** One episode per message? Per session? Per topic? Our code defaults to one episode per user turn (`ingest.py:142-150`) plus optional derived-finding episodes. No published guidance. Anthropic Dreams uses entire sessions. The DMR benchmark used messages. **We should ablate** in P0.6.

3. **Whether community detection helps retrieval for agent memory specifically.** Microsoft GraphRAG [13] reports lift for *document* corpora. Zep paper [1] uses label propagation, not Leiden, and does not separately ablate communities in DMR/LongMemEval. **Unknown.** Our recommendation is to skip in P0, revisit if relevance plateaus.

4. **Multi-tenant scaling beyond ~16k users on FalkorDB.** Redis Cluster's hash slot limit. No published reference for sharding Graphiti across FalkorDB clusters. We should plan a hash-by-user-id shard map for the eventual transition.

5. **Cost-per-quality tradeoff between Sonnet and Haiku for extraction.** Graphiti is LLM-bound. Our `config.py:38` defaults to `gpt-4.1-mini`; upstream defaults to GPT-4. Neither has published per-quality cost analysis. Run for one week with each, compare resolution/extraction quality, choose.

6. **Embedding model drift.** Our `text-embedding-3-small` (`config.py:51`) bonds existing edges' `fact_embedding` to that model. If OpenAI deprecates or replaces it, we either re-embed all edges (expensive) or accept distribution drift. No published guidance. The TempEL paper [31] is the closest analog — distribution drift over 9 years cost ~3-18% accuracy.

7. **Concurrent `add_episode` semantics under failure.** What happens if extraction succeeds, edge insert succeeds, but community-update fails halfway? Graphiti's transactionality story is undocumented. Our code wraps in `try/except` (`ingest.py:81-92`) and logs — that's all we can do.

8. **Long-tail entity behavior.** Graphiti's resolver works well when entities are mentioned often enough to develop a stable representation. Rare entities (one-shot mentions) frequently get duplicated. No mitigation in the literature; possibly an LLM-resolver tightening per [10].

9. **Whether "dreaming"-style consolidation actually improves agent behavior.** Anthropic's *Dreams* feature [Anthropic Managed Agents docs, 2026] reports Harvey 6x completion-rate lift, but no controlled study. Our dream pass is implementing on this thin empirical base. The benchmark harness (P0.6) is the gate that catches whether we are net-positive.

---

## 10. Recommended reading order

If you have time for three things, read in order:

1. **The Zep paper [1].** ~12 pages, 30 minutes. Single source of truth for the architecture you are calling APIs against. Has the benchmark numbers worth knowing.
2. **Snodgrass-via-Fowler [Fowler 2021, martinfowler.com/articles/bitemporal-history.html].** ~15 minutes. The whole "valid time vs transaction time" distinction is load-bearing for the dream pass and easy to get wrong.
3. **Generative Agents [23] §4 (Memory).** ~30 minutes. The relevance × recency × importance scoring is the reference for any custom reranking we layer over Graphiti's search.

If you have a second tier, read:

4. **Microsoft GraphRAG [13]** for the global-sensemaking / community-summary angle. Helps justify our "no communities in P0" call.
5. **HippoRAG [17]** for the complementary-learning-systems framing of agent memory. Useful background for the dream pass design.
6. **Sleep-time Compute [25]** for the precomputation-during-idle framing of P4 (scenario pre-warm).
7. **CoALA [26]** for the working/episodic/semantic/procedural memory taxonomy. Sanity check on our `MemoryKind` enum.

If you have a deep dive day:

8. **Graphiti README + CHANGELOG [2, 3]**.
9. **Graphiti docs core-concepts pages [4, 5, 6, 7].**
10. **Zep blog [9, 10, 11].**

---

## References

[1] Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., Chalef, D. "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv:2501.13956, Jan 2025. https://arxiv.org/abs/2501.13956 — HTML: https://arxiv.org/html/2501.13956v1

[2] getzep/graphiti GitHub repository. https://github.com/getzep/graphiti

[3] Graphiti release notes (v0.27.0 through v0.29.0). https://github.com/getzep/graphiti/releases

[4] Zep Docs — Core Concepts: Adding Episodes. https://help.getzep.com/graphiti/core-concepts/adding-episodes

[5] Zep Docs — Core Concepts: Communities. https://help.getzep.com/graphiti/core-concepts/communities

[6] Zep Docs — Core Concepts: Custom Entity and Edge Types. https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types

[7] Zep Docs — Working with Data: Searching. (Note: URL was a 404; content reconstructed via secondary search.) https://help.getzep.com/v2/searching-the-graph

[8] Zep Docs — Working with Data: Adding Fact Triples. https://help.getzep.com/graphiti/working-with-data/adding-fact-triples

[9] Zep Blog — "Beyond Static Knowledge Graphs: Engineering Evolving Relationships." https://blog.getzep.com/beyond-static-knowledge-graphs/

[10] Zep Blog — "LLM Data Extraction at Scale." https://blog.getzep.com/llm-rag-knowledge-graphs-faster-and-more-dynamic/

[11] FalkorDB Blog — "Graphiti + FalkorDB: Integration for Multi-Agent Systems." https://www.falkordb.com/blog/graphiti-falkordb-multi-agent-performance/

[12] DeepWiki — getzep/graphiti architecture overview. https://deepwiki.com/getzep/graphiti

[13] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., Larson, J. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130, Microsoft Research, Apr 2024. https://arxiv.org/abs/2404.16130

[14] FalkorDB Blog — "Graphiti: FalkorDB support and 14K GitHub Stars." Zep blog version: https://blog.getzep.com/graphiti-knowledge-graphs-falkordb-support/

[15] FalkorDB Documentation — Graphiti integration. https://docs.falkordb.com/agentic-memory/graphiti.html

[16] Guo, Z. et al. "LightRAG: Simple and Fast Retrieval-Augmented Generation." arXiv:2410.05779, Oct 2024 (EMNLP 2025). https://arxiv.org/abs/2410.05779

[17] Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." NeurIPS 2024. https://arxiv.org/abs/2405.14831

[18] McClelland, J. L., McNaughton, B. L., O'Reilly, R. C. "Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory." Psychological Review 102(3), 1995. https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf

[19] Kumaran, D., Hassabis, D., McClelland, J. L. "What learning systems do intelligent agents need? Complementary learning systems theory updated." Trends in Cognitive Sciences 20(7), 2016. https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30043-2

[20] Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Ni, L., Shum, H.-Y., Guo, J. "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph." ICLR 2024 (arXiv:2307.07697). https://arxiv.org/abs/2307.07697

[21] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020 (arXiv:2005.11401). https://arxiv.org/abs/2005.11401

[22] Sanmartin, D. "KG-RAG: Bridging the Gap Between Knowledge and Creativity." arXiv:2405.12035, May 2024. https://arxiv.org/abs/2405.12035

[23] Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., Bernstein, M. S. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023 (arXiv:2304.03442). https://arxiv.org/abs/2304.03442

[24] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., Gonzalez, J. E. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, Oct 2023. https://arxiv.org/abs/2310.08560

[25] Lin, K., Snell, C., Wang, Y., Packer, C., Wooders, S., Stoica, I., Gonzalez, J. E. "Sleep-time Compute: Beyond Inference Scaling at Test-time." arXiv:2504.13171, Apr 2025. https://arxiv.org/abs/2504.13171

[26] Sumers, T. R., Yao, S., Narasimhan, K., Griffiths, T. L. "Cognitive Architectures for Language Agents (CoALA)." TMLR 2024 (arXiv:2309.02427). https://arxiv.org/abs/2309.02427

[27] Snodgrass, R. T. "Developing Time-Oriented Database Applications in SQL." Morgan Kaufmann, 1999. http://www2.cs.arizona.edu/~rts/publications.html — canonical reference for bi-temporal SQL. SQL:2011 standard mirrors the vocabulary.

[28] Allen, J. F. "Maintaining Knowledge about Temporal Intervals." Communications of the ACM 26(11), 1983. https://cse.unl.edu/~choueiry/Documents/Allen-CACM1983.pdf

[29] Cai, B., Xiang, Y., Gao, L., Zhang, H., Li, Y., Li, J. "Temporal Knowledge Graph Completion: A Survey." IJCAI 2023. https://www.ijcai.org/proceedings/2023/734 — arXiv preprint: https://arxiv.org/abs/2201.08236

[30] Qian, X. et al. "TimeR4: Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering." EMNLP 2024. https://aclanthology.org/2024.emnlp-main.394/

[31] Zaporojets, K., Kaffee, L.-A., Deleu, J., Demeester, T., Develder, C., Augenstein, I. "TempEL: Linking Dynamically Evolving and Newly Emerging Entities." NeurIPS 2022 Datasets & Benchmarks (arXiv:2302.02500). https://arxiv.org/abs/2302.02500

[32] Hogan, A., Blomqvist, E., Cochez, M., et al. "Knowledge Graphs." ACM Computing Surveys 54(4), Article 71, 2021. https://dl.acm.org/doi/10.1145/3447772 — arXiv preprint: https://arxiv.org/abs/2003.02320

[33] Graphiti issue tracker. https://github.com/getzep/graphiti/issues — issues #760, #871, #893, #1001, #1193, #1299, #1438, #1440, #1441, #1452, #1465, #1467, #1469, #1470, #1477, #1481, #1483 referenced throughout.

[34] Angles, R., Gutiérrez, C. "Survey of graph database models." ACM Computing Surveys 40(1), 2008. https://dl.acm.org/doi/10.1145/1322432.1322433

[35] Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. "A-MEM: Agentic Memory for LLM Agents." NeurIPS 2025 (arXiv:2502.12110). https://arxiv.org/abs/2502.12110

[36] Konda, P., Das, S., Suganthan, P., Doan, A., Ardalan, A., Ballard, J. R., Li, H., Panahi, F., Zhang, H., Naughton, J., Prasad, S., Krishnan, G., Deep, R., Raghavendra, V. "Magellan: Toward Building Entity Matching Management Systems." Proceedings of VLDB Endowment 9(12), 2016. http://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf

[37] Christen, P. "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection." Springer, 2012. https://users.cecs.anu.edu.au/~Peter.Christen/data-matching-book-2012.html

[38] Graphiti issue #760 — gpt-4.1-nano dedup hallucination. https://github.com/getzep/graphiti/issues/760

---

End of report. Total length ~9,500 words.
