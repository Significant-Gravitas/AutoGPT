# Multi-Episode Memory / Schema Integration — Design Doc

Roadmap item: **P12 — Schema integration / multi-episode summaries** (TODO.md §P12).

This document is the design and build plan for durable consolidated-summary nodes
that survive across many dream passes. It is the cognitive-science / data-model
half of the dream system: P0 ships the dream pass that emits flat episodes; P12
gives those passes a place to *deposit lasting structure*, so "what I know about
Project X" condenses without losing provenance.

The document is long because the problem is the hardest item on the dream
roadmap and the team will commit to it on the strength of this writeup. Skim the
**Decision** subsection at the end of each major section; the prose is there
because the wrong answer is expensive to undo.

---

## 0. Decisions in one screen

1. **Build a custom `:Summary` node type in FalkorDB.** Reject Graphiti
   `:Community` (full-rebuild, no stable identity) and reject moving summaries
   to Postgres (loses the graph join with entities/episodes).
2. **Identity = `(group_id, scope, summary_kind, subject_key)`, versioned via
   `:SUPERSEDES`-chains.** Never overwrite. Each dream emits a new
   `:Summary` node and points it at the prior version for that key.
3. **Two summary kinds at v1**: `entity_summary` (per subject — person,
   project, book) and `scope_summary` (per scope — `real:global`,
   `project:billing`, etc.). Reject the full RAPTOR-style recursive tree for
   v1; add `meta_summary` (summary-of-summaries) at v2 only if metrics demand
   it.
4. **Mem0-style write loop on every dream**: for each candidate subject, the
   pass decides `CREATE` / `UPDATE` / `NOOP` / `RETIRE`. `UPDATE` writes a new
   `:Summary` linked back to the previous version; `RETIRE` writes a tombstone
   summary so retrieval stops returning the subject's prior summary chain.
5. **Retrieval tier**: warm-context renders `:Summary` content *above*
   relationship facts and recent episodes. Summary tier is a third format
   alongside `<FACTS>` and `<RECENT_EPISODES>` — not a replacement.
6. **Provenance is mandatory.** Every `:Summary` carries `source_episode_uuids`,
   `source_summary_uuids`, `generation`, `supersedes_uuid`, `provenance` (the
   dream job id), and a content hash. We pay the storage cost.
7. **Build in three phases over ~3 engineer-weeks**: (P12a) schema +
   write/read primitives, (P12b) wire into dream pass + warm context, (P12c)
   meta-summaries + cross-scope summaries.
8. **Eval is gated on existing P0.6 benchmark harness** plus three new
   summary-specific metrics: stability (regeneration cosine), provenance
   recall, and compression ratio. Dedicated golden set per summary kind.

The rest of this document defends each of those decisions and tells the
implementer where every line of code goes.

---

## 1. Why we need this — restate the problem precisely

Today (after P0 lands) the dream pass writes back two things to Graphiti:

- **Consolidated facts**: one short string per fact, written as a JSON
  `MemoryEnvelope` via `enqueue_episode()`. Graphiti's entity-extraction LLM
  pulls entities and emits `:RELATES_TO` edges between them.
- **Demotions / status flips**: `expired_at` set on contradicted edges, or a
  custom `status='superseded'` property added by the not-yet-written
  `mark_edges_superseded` helper (`dreaming-memory.md` §8 Priority 2).

What is missing is a place to put consolidated *prose* that the dream pass
believes is durably useful and that should survive across nights. Concretely:

- The user has had 14 conversations about Project Billing over six weeks.
  Tonight's dream identifies seven facts and one preference. Last night's
  dream identified six of those seven facts and a different preference.
  Without a `:Summary` node, the only durable trace is the seven episodes and
  some edges; there is no single object that says "as of 2026-05-12, the user
  thinks about Project Billing this way." The next chat session has no place
  to retrieve such an object from. Warm context just gets the facts again.
- When the user asks a fresh assistant "what do you know about Project
  Billing", today the only assembly path is `client.search(query="Project
  Billing")` → relationship facts → render. There is no condensed paragraph
  the assistant can drop into its system prompt that captures intent,
  preferences, stakeholders, sticking points, and recent decisions in one
  block, with provenance.
- The dream pass currently has no way to *learn from its own prior outputs*.
  Each night starts over against the raw episode store. Two-month-old
  consolidated facts get re-derived from the same raw episodes, costing
  tokens; whatever the dream "figured out" last week is lost the moment its
  output episodes scroll out of the 50-episode window.

The cognitive-science framing for what we are building is exactly *systems
consolidation* — what McClelland, McNaughton, and O'Reilly call complementary
learning systems (1995): hippocampal episodic traces get reinstated over time
and gradually deposit changes in neocortical schemas that are no longer tied
to single-event provenance ([Stanford
PDF](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)). Tse et
al.'s 2007 *Science* paper on rapid neocortical learning showed that when an
incoming memory is *schema-consistent*, it can be incorporated into
neocortical structure within a single day rather than the weeks-to-years
canonical timeline ([PubMed](https://pubmed.ncbi.nlm.nih.gov/17412951/)).

We are not claiming neuroscience-grade fidelity. We are claiming that the
right software shape for this problem is a tier of nodes that are *durable*,
*regeneratable*, *provenance-preserving*, and *not the same as either raw
episodes or relationship facts*. That tier is the `:Summary` node.

---

## 2. Prior art — what to copy, what to reject

This section is long because P12's value depends on stealing from the right
prior art. Each subsection ends with one or two lines on **what we copy** and
**what we reject**.

### 2.1 Graphiti `:Community` nodes (upstream — already in our schema, unwired)

Graphiti-core 0.28.2 ships a complete community-detection subsystem:

- `Graphiti.build_communities(group_ids)` runs label propagation over the
  entity graph (`utils/maintenance/community_operations.py:217`), produces
  `:Community` nodes summarising each cluster, and writes `:HAS_MEMBER` edges
  to member entities ([upstream
  source](https://github.com/getzep/graphiti)). We have it installed; we do
  not call it. Confirmed by `grep -rn build_communities` returning only the
  AGENTS.md cookbook reference.
- The summaries are written by **pairwise binary-tree merge** of entity
  `summary` strings: the operation peels off odd entries, pairs the rest,
  asks an LLM to `summarize_pair`, then loops until one survives
  (`community_operations.py:178-200`). Then it asks the LLM for a
  `generate_summary_description` on the result and stores it as the
  community's `name`. This is essentially how Wu et al.'s OpenAI book
  summarization paper works on prose
  ([arXiv:2109.10862](https://arxiv.org/abs/2109.10862)), but driven by
  entity descriptions rather than text chunks.
- **The fatal flaw for our use case**: `Graphiti.build_communities` first
  calls `remove_communities(driver)` which executes `MATCH (c:Community)
  DETACH DELETE c`. Every run is a full rebuild. No stable identity, no
  provenance chain from yesterday's community to today's, no history.
  See `community_operations.py:245-257`.
- Zep — the company behind Graphiti — has fixed this in their cloud product
  but **not in the open-source library at our version**. The Zep paper
  describes their three-tier subgraph (episode / entity / community), where
  community updates use "a single recursive step in label propagation" when
  a new entity arrives and the community summary is updated incrementally
  ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956)). They still need
  periodic full rebuilds because "resulting communities gradually diverge
  from those that would be generated by a complete label propagation run."
- Graphiti's search recipes have a `CommunitySearchConfig`
  (`search/search_config_recipes.py:49`) that participates in hybrid
  retrieval via BM25 + cosine similarity over community `name`/`name_embedding`,
  with RRF / MMR / cross-encoder rerankers. So if we *did* populate
  `:Community` nodes, our existing `client.search()` would surface them.
  But we'd need to use `client.search_()` (the advanced variant) with a
  config that includes `community_config` — `client.search()` currently
  returns only `:RELATES_TO` edges.

**Copy**: the label-propagation clustering, the `:HAS_MEMBER`-style membership
edge, the integration into hybrid search.

**Reject**: the destroy-and-rebuild semantics, the lack of typed metadata
(no `kind`, no `status`, no `confidence`, no `supersedes_uuid`), the lack of
provenance from each community back to specific episodes. We will write a new
node label rather than try to subclass `:Community` — the upstream API does
not expose enough seams, and we want the freedom to evolve the schema
without forking graphiti-core.

### 2.2 MemGPT / Letta — archival memory and sleep-time agents

MemGPT introduced the "virtual context management" framing
([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)). The relevant
architecture for us is the tiering — **core memory** (in-context, editable
memory blocks pinned in the prompt), **recall storage** (full conversation
history searchable by tool), **archival storage** (an external vector
database the agent reads and writes via tools)
([Letta docs](https://docs.letta.com/concepts/memgpt/),
[Letta blog](https://www.letta.com/blog/agent-memory)). Letta's later
**sleep-time agents** add a second agent that runs in the background and is
the *only* actor with permission to edit the primary agent's core memory
blocks ([Letta sleep-time blog](https://www.letta.com/blog/sleep-time-compute)).

A few things we should explicitly steal:

- **Memory blocks have a schema**: `label`, `description`, `value`,
  `character_limit`. The `description` exists so the agent knows *what kind of
  thing* to put in the block. This is conceptually identical to what our
  `summary_kind` field will be.
- **The asymmetric write model**: primary agent reads, sleep-time agent
  writes. We have the same asymmetry — the chat session reads `:Summary`
  nodes (via warm context); only the dream pass writes them. This avoids
  the race conditions that `dreaming-memory.md` §6 spent a page worrying
  about, because the chat path never mutates summaries.
- **The decision verbs**: Mem0 (an adjacent product to MemGPT) makes the
  update mechanism explicit. Their `Update Phase` retrieves top-k similar
  existing memories and asks the LLM to pick `ADD` / `UPDATE` / `DELETE` /
  `NOOP` per candidate ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)).
  This is the cleanest verb set in the agentic-memory literature; we'll
  borrow it verbatim, renaming `DELETE` to `RETIRE` to avoid confusion with
  hard delete.

**Copy**: typed memory blocks with `label`/`description`/`value`, sleep-time-
agent write model, Mem0's ADD/UPDATE/NOOP/RETIRE verbs.

**Reject**: in-context "block" framing as a primary mental model — our summaries
live in the graph, not in the system prompt, and we render them into the
prompt only at retrieval time. Also reject: tying summary lifecycle to the
agent's tool calls. Our dream pass is scheduled, not agent-driven.

### 2.3 Generative Agents (Park et al., Stanford, 2023)

The memory-stream + reflection architecture
([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)) is the cleanest
academic prior art for a system that takes raw episodes and synthesizes
higher-level conclusions on a periodic basis:

- Every memory carries content text, created-at timestamp, last-accessed
  timestamp, and an importance score (LLM-rated 1-10 by direct prompt).
- Reflections trigger when "the sum of the importance scores for the latest
  events perceived by the agents exceeds a threshold (150 in our
  implementation)" — empirically about 2-3 times per simulated day.
- The reflection algorithm: query the LLM with the 100 most recent memories
  to generate candidate questions, retrieve relevant memories for each
  question, ask the LLM for 5 high-level insights with evidence citations,
  store insights as new memories. Reflections can cite other reflections,
  building a hierarchy.
- Retrieval scores each memory by `α_recency * recency + α_importance *
  importance + α_relevance * relevance`, all α=1, recency exponentially
  decayed (0.995 per sim hour), relevance = cosine similarity of embeddings.

What's notable for us: **the paper has no mechanism to prevent redundant
reflections.** Each reflection round potentially produces summaries
overlapping with prior summaries; the system relies on retrieval-time
relevance scoring to surface the "right" one. They got away with this in a
two-day sandbox simulation; we cannot get away with it in a six-month
production system.

**Copy**: the importance score (we'll generalize to `confidence` per
`MemoryEnvelope`, which we already have), reflections citing reflections (our
`source_summary_uuids` field), evidence-grounded synthesis.

**Reject**: rebuild-every-time-a-threshold-trips. We will use stable
identity to suppress redundant reflections.

### 2.4 MemoryBank (Zhong et al., 2023)

MemoryBank ([arXiv:2305.10250](https://arxiv.org/abs/2305.10250)) is the
paper most often cited for "Ebbinghaus-curve-based forgetting in LLM
memory." For our purposes the relevant ideas are:

- Each memory has a *memory strength* `S` that decays exponentially with
  time since last access and is reinforced on each retrieval —
  `S = e^(-t/S_prev)` is the textbook Ebbinghaus form (the paper uses a
  variant). This composes naturally with our existing `confidence` field
  and provides a principled forgetting curve.
- A *user portrait* (the paper's term for what we'd call a self-model
  summary) is constructed on a slow cadence by summarizing personality
  signals from past interactions. This is the analog of our future
  `scope_summary` for `real:global`.

**Copy**: the *idea* that decay belongs as a per-memory scalar rather than a
binary status flip. P0 already operates on `MemoryStatus` (active / tentative
/ superseded / contradicted) — that's a binary. We will add a decay-aware
`relevance_score` on `:Summary` nodes (recency boost) without changing the
status enum.

**Reject**: making forgetting central. Our hard requirement is provenance,
which means we don't really *forget* — we mark superseded and stop returning
in default search. Decay is a relevance signal, not a deletion signal.

### 2.5 LangChain `ConversationSummaryMemory` / `ConversationSummaryBufferMemory`

LangChain's classic summary memory keeps a running natural-language summary
of the conversation; on each turn it re-summarizes by feeding the prior
summary + the new turn into an LLM. `ConversationSummaryBufferMemory` keeps a
recent-turn buffer plus a summary of older turns, re-summarizing when the
buffer hits a token threshold. Both are now deprecated in favor of
LangGraph's persistence primitives
([migrating-memory docs redirect](https://docs.langchain.com/oss/python/langchain/overview)).

This is the *simplest* possible regeneration pattern: one global summary,
overwritten every time, no version chain, no identity beyond "the
conversation." It's the cautionary tale: by the third re-summarization, the
summary contains errors that no longer trace to any source. We are
explicitly building the opposite of this.

**Copy**: nothing structural. But the failure mode — *recursive
summarization compounds errors at a measurable rate* — is the single most
important risk to design against. See §7 (Failure modes).

**Reject**: implicit identity, in-place updates, no provenance.

### 2.6 LlamaIndex — `SummaryIndex`, `TreeIndex`, `DocumentSummaryIndex`, `ComposableGraph`

LlamaIndex offers four indexes that are conceptually adjacent
([API ref](https://developers.llamaindex.ai/python/framework-api-reference/indices/tree/)):

- **SummaryIndex** (formerly ListIndex): a flat list of chunks; queries do
  tree-summarize over retrieved nodes at *query time*, not at indexing time.
- **TreeIndex**: builds a hierarchical tree of summaries *at index time*. Each
  internal node is a fixed-K-ary summary of its children. Retrieval modes:
  traverse top-down (route the query down the tree) or "tree summarize"
  (gather candidates and bottom-up summarize again).
- **DocumentSummaryIndex**: stores one summary per document plus all chunks;
  retrieval matches against summaries to pick documents, then fetches
  full chunks. This is the cleanest match for what we want at the
  `entity_summary` level — one durable summary per subject, with the
  chunks/episodes still retrievable.
- **ComposableGraph**: compose multiple indexes into a tree. You can have a
  per-document `SummaryIndex` per project, then a top-level `TreeIndex` over
  those.

The pattern we're stealing is `DocumentSummaryIndex`: one summary per
addressable subject, summary stored separately from but linked to its source
content. We're going further — versioned, provenance-preserving, and graph-
native — but the topology matches.

**Copy**: one-summary-per-subject with chunk-level fallback. The retrieval
flow "match against summary, fall back to source episodes for detail" is
exactly what we want when a user asks a specific question about a subject.

**Reject**: fixed tree topology, query-time summarization, treating
re-summarization as cheap.

### 2.7 RAPTOR (Sarthi et al., 2024)

RAPTOR ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)) is the
strongest single piece of prior art on *retrieval-quality* terms. The build
recipe:

- Embed every chunk (100 tokens).
- UMAP to lower dimension; GMM (with BIC for model selection) for **soft**
  clustering — a chunk can belong to multiple parent clusters.
- LLM-summarize each cluster (gpt-3.5-turbo in the paper, with the prompt
  "summarize including as many key details as possible").
- Recurse: treat the summary embeddings as the next level's chunks.
- Average cluster size ≈ 6.7; tree depth up to 5 for long documents.
- Reported summary-to-input compression: **0.28** (72% reduction).
- Hallucination rate during summarization: **~4%**, but the authors observe
  these did not propagate up to parent nodes (they got smoothed out).
- Retrieval: two modes; **collapsed tree** (treat all nodes as a flat list,
  search over everything, retrieve top-k by score) outperformed tree
  traversal on QASPER subset evaluations.
- Headline result: +20 points absolute on QuALITY with GPT-4 reader.

The biggest reason RAPTOR works is **soft clustering** — when a chunk
genuinely belongs to two topics, GMM lets it sit in both summary chains. A
hard-clustering approach (k-means, single-membership label propagation)
forces a choice and either loses detail or creates duplicate summaries.

For our setting RAPTOR is more relevant as inspiration than as drop-in:
- We don't have a static document; we have an episode stream that grows
  every day. RAPTOR's tree is built once.
- We need stable identity across rebuilds. RAPTOR rebuilds from scratch each
  time the source corpus changes.
- We care about per-subject summaries (a person, a project), not arbitrary
  semantic clusters. Soft clustering is overkill when the unit of summary is
  "Sarah" or "project:billing".

**Copy**: the 0.28 compression ratio as a *target* for our `entity_summary`
sizing; the collapsed-tree retrieval finding (it argues against ranking by
hierarchy level, which is good news because we don't want to either); the
~4% hallucination rate as a metrics-baseline for our stability eval.

**Reject**: GMM/UMAP/soft clustering — we'll cluster by Graphiti entity
identity for entity summaries and by scope string for scope summaries. We'll
defer the recursive-tree case to v2 (meta-summaries).

### 2.8 HippoRAG (Gutiérrez et al., NeurIPS 2024)

HippoRAG ([arXiv:2405.14831](https://arxiv.org/abs/2405.14831)) is
explicitly hippocampus-inspired. The architecture:

- LLM OpenIE turns passages into noun-phrase nodes and labeled relation
  edges. (We already do this — Graphiti runs LLM entity extraction.)
- Synonymy edges between entities are added when retrieval-encoder cosine
  similarity exceeds 0.8.
- Query → entities → seed nodes for Personalized PageRank (damping 0.5).
  Probability mass flows through the neighborhood of relevant nodes.
- Single-step PPR matches or beats iterative retrieval (IRCoT) on
  multi-hop QA — and is reported to be 10-30x cheaper and 6-13x faster.
- The paper compares directly against RAPTOR: HippoRAG **40.9-71.5% R@2**
  across MuSiQue / 2WikiMultiHop / HotpotQA vs RAPTOR's **35.7-46.3% R@2**.
  Graph-based wins for multi-hop.

HippoRAG is the closest evidence we have that **graph-native consolidation
beats hierarchical-summary consolidation** on multi-hop questions. That maps
to our setting: when the user asks "how is project billing connected to
Sarah's transition?", the answer comes from edges between entities, not
from any single summary. Summaries are good for "what do I know about Sarah"
or "what's project billing about"; PPR-style traversal is good for the
relational questions.

**Copy**: the *philosophy*. Summaries are not a replacement for the graph;
they sit on top of it. Our `:Summary` node has explicit edges to the
entities it covers, so HippoRAG-style retrieval becomes available later.

**Reject**: replacing our retrieval stack with PPR for now. Graphiti's
hybrid search already does what we need at warm-context volume.

### 2.9 Cognitive science: Complementary Learning Systems, schema theory

- **McClelland, McNaughton, O'Reilly (1995)** — *Why there are
  complementary learning systems in the hippocampus and neocortex*
  ([Stanford PDF](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)).
  Hippocampus stores discrete, sparse, pattern-separated episodic traces.
  Neocortex stores distributed, overlapping representations that integrate
  across episodes to produce schema. *Reinstatement* of recent
  hippocampal traces interleaves them with neocortical replay; neocortex
  changes a little on each reinstatement. Remote memory is accumulated
  neocortical change.
- **Tse et al. (2007)** — rats with prior food-place schemas in a paddock
  learned new food-place associations *in a single day* (vs. weeks for
  schema-naïve animals). Schema-consistency dramatically accelerates
  systems consolidation
  ([PubMed](https://pubmed.ncbi.nlm.nih.gov/17412951/)).
- **Bartlett (1932)** — schema as "active organization of past reactions";
  memory is reconstructive rather than reproductive.

What this gives us, design-wise:

1. The right unit of consolidation is *the schema*, not *the episode*. Our
   summary nodes correspond to neocortical schemas, not to specific events.
2. Schemas update *incrementally* over many reinstatement events. We should
   not rebuild a summary from scratch when only a few episodes have changed
   — we should let the dream pass *amend* the prior summary, citing both
   the old summary and the new episodes.
3. Schema-consistent new facts should integrate fast; schema-violating
   facts should trigger a noisier rewrite. This is exactly the Mem0
   ADD/UPDATE/RETIRE distinction.

**Copy**: incremental update semantics; schema-as-organizing-unit;
schema-consistency as a write-effort modulator. (Operationally: if the new
episodes are 95%+ consistent with the prior summary, an UPDATE is a small
patch; if 50%+ contradicted, it's effectively a CREATE-from-scratch.)

**Reject**: any pretense that we are doing cognitive simulation. This is
software inspired by an analogy.

### 2.10 Microsoft GraphRAG

GraphRAG ([arXiv:2404.16130](https://arxiv.org/html/2404.16130v2),
[microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/)) is
the most production-shaped of the graph + hierarchical summary approaches:

- Indexing: 600-token chunks → entity / relationship / claim extraction →
  graph. Then **hierarchical Leiden** clustering. Recursive — keep
  subdividing until communities are small enough.
- Per-community LLM-generated report with structured schema: title,
  executive summary, impact severity (0-10), rating explanation,
  detailed findings (5-10 insights, each with summary + paragraph + data
  citations).
- Two query modes: **global** (map-reduce over all community summaries,
  filter on per-community helpfulness score, reduce to final answer);
  **local** (entity neighborhood expansion).
- Reported indexing cost: 1M tokens → 281 minutes on 16GB VM using
  GPT-4-turbo (mid-2024). Context-window usage: lowest level (C0) 26,657
  tokens, highest level (C3) 746,100 tokens. Source-text baseline 1,014,611
  tokens. So C0 achieves 97% reduction with 72% comprehensiveness win rate
  vs vector RAG.
- Graph stats: 8,564 nodes / 20,691 edges / 34 root communities on the
  podcast benchmark.

**Copy**: structured report schema (title / summary / findings, each with
citations). The "findings list with per-finding citations" pattern is what
our `:Summary` content field should look like — not a single flat paragraph
but a structured body.

**Reject**: hierarchical Leiden as the only clustering signal. Our scopes
already give us a natural top-level partition (`project:*`, `book:*`,
`real:global`). We don't need to discover them via clustering — we can use
them directly.

### 2.11 Anthropic *Dreams* (Managed Agents)

Documented in `dreaming-anthropic.md`. Two design choices to copy and one to
explicitly reject:

- **Copy: immutable inputs, diffable outputs.** A dream reads memory + sessions,
  produces a *new* memory store. The input is never mutated
  ([docs](https://platform.claude.com/docs/en/managed-agents/dreams)). For
  us, this means: dream-pass reads the current `:Summary` set for the user
  but writes new `:Summary` nodes; the prior set is preserved by
  `:SUPERSEDES` chains.
- **Copy: chain-of-dreams.** The Anthropic output store becomes the next
  pass's input store, creating an audit-trail chain. We mirror this — the
  latest dream's `:Summary` set is yesterday's input for tomorrow's
  consolidation.
- **Reject: take-it-or-leave-it replacement.** Anthropic's output is a whole
  new store; the operator attaches it or discards it. For us, every
  individual summary is independently superseded/preserved — finer-grained
  identity than Anthropic's "swap the whole thing" model. This is also the
  shape that gives us per-summary user veto (P7) for free.

---

## 3. The data model

This is the central deliverable. Spec is targeted at FalkorDB / Graphiti
v0.28.2 as currently used.

### 3.1 Node label: `:Summary`

A new node label, peer to `:Entity`, `:Episodic`, `:Community`. Stored in
the same per-user FalkorDB database (`group_id`).

| Property | Type | Required | Description |
|---|---|---|---|
| `uuid` | string | yes | UUIDv4. Primary key. |
| `group_id` | string | yes | Same as `derive_group_id(user_id)`. Indexed. |
| `scope` | string | yes | `real:global`, `project:foo`, `book:bar`, `session:S`. Indexed. |
| `summary_kind` | string | yes | Enum: `entity_summary` \| `scope_summary` \| `meta_summary`. Indexed. |
| `subject_key` | string | yes | Stable subject identifier — see §3.3. Indexed (composite with `summary_kind`, `scope`). |
| `subject_uuid` | string | no | For `entity_summary`, the `:Entity.uuid` of the subject. Null for `scope_summary` / `meta_summary`. |
| `title` | string | yes | One-line human-readable title. |
| `content` | string | yes | Structured prose body — see §3.4. |
| `content_hash` | string | yes | SHA-256 of `content`. Used to short-circuit `NOOP` updates. |
| `content_embedding` | vector | yes | Embedding of `title + content`. Used for retrieval and stability eval. |
| `confidence` | float | yes | 0.0–1.0. LLM-self-rated by the dream pass. |
| `status` | string | yes | Enum from `MemoryStatus`: `active` \| `tentative` \| `superseded` \| `contradicted`. |
| `generation` | int | yes | Monotonic counter per `subject_key`. v1 = 1, v2 = 2, ... |
| `provenance` | string | yes | `dream:{job_id}:{phase}` (phase ∈ `phase1`, `phase2`, `manual`). |
| `created_at` | datetime | yes | Wall-clock at write. |
| `valid_from` | datetime | yes | Effective-from. Usually = `created_at`. |
| `valid_to` | datetime | nullable | Effective-until. Null = current. Set when superseded. |
| `supersedes_uuid` | string | nullable | UUID of the prior version this replaces. Null for `generation=1`. |
| `superseded_by_uuid` | string | nullable | Set when this summary is replaced. Backward link. |
| `source_episode_uuids` | list[string] | yes | UUIDs of `:Episodic` nodes that contributed evidence. May be empty for `meta_summary`. |
| `source_summary_uuids` | list[string] | yes | UUIDs of prior `:Summary` nodes that were merged or amended. |
| `source_fact_uuids` | list[string] | yes | UUIDs of `:RELATES_TO` edges that were summarized. |
| `instruction_hash` | string | nullable | Hash of the dream-pass instructions (P12 §6.4 — for stability). |

`uuid` and `group_id` follow Graphiti conventions. `scope`, `summary_kind`,
`subject_key`, and `status` are all FalkorDB-indexable strings — for v1
we'll add explicit `CREATE INDEX` calls in the boot path.

`content_hash` lets the dream pass short-circuit: if it regenerates a
summary that hashes identically to the current `active` one, emit `NOOP` and
do not write a new node. This is the single biggest cost-saver against
LangChain-style "re-summarize on every run."

### 3.2 Edges

| Edge label | From | To | Properties | Purpose |
|---|---|---|---|---|
| `:SUPERSEDES` | `:Summary (new)` | `:Summary (old)` | `group_id`, `created_at`, `reason` | Version chain. |
| `:COVERS` | `:Summary` | `:Entity` | `group_id`, `created_at`, `weight` (0.0-1.0, optional) | For `entity_summary`, the entity covered. For `scope_summary`, every key entity in the scope (capped at top N by mention count). |
| `:DERIVED_FROM` | `:Summary` | `:Episodic` | `group_id`, `created_at` | Provenance link to source episodes. Redundant with `source_episode_uuids` but lets us run Cypher graph queries efficiently. |
| `:MERGED_FROM` | `:Summary (new)` | `:Summary (old)` | `group_id`, `created_at` | Used when a `meta_summary` summarizes multiple `:Summary` nodes. |
| `:CONTRADICTS` | `:Summary` | `:Summary` | `group_id`, `created_at`, `evidence_episode_uuid` | Used by the dream pass when a new summary explicitly contradicts an old one (vs. just supersedes). |

`:SUPERSEDES` is the load-bearing edge. Walking it backward from any summary
gives you the full provenance chain. Walking forward (via
`superseded_by_uuid` on the property side, or the inverse `:SUPERSEDES`
edge) gives you "the current version of this summary."

### 3.3 Subject keys

A subject key is the *stable identity slot* across regenerations. It must be
deterministic — two dream passes that pick the same subject key for the
same conceptual thing must always produce updates that link via
`:SUPERSEDES`, not parallel chains.

| `summary_kind` | `subject_key` format | Example |
|---|---|---|
| `entity_summary` | `entity:{entity_uuid}` | `entity:a3f...c91` |
| `scope_summary` | `scope:{scope_string}` | `scope:project:billing-app` |
| `meta_summary` | `meta:{deterministic_hash}` (v2) | `meta:8f01...` |

The `entity_uuid` is Graphiti's UUID for the entity node. Graphiti's
entity-resolution is good but not perfect — if the LLM extraction creates a
duplicate "Sarah" entity, we'd write a parallel summary chain.
**Mitigation**: the dream pass's CREATE step does an explicit "is this
entity an alias of an existing entity covered by an active summary?" check
before assigning a fresh subject key. We piggy-back on the existing dedup
work happening in P2 — when P2 ships, summary-creation uses the same
similarity threshold (`fact_embedding` cosine > 0.92) to decide whether to
attach to an existing chain or start a new one.

For `scope_summary`, the key is just the scope string verbatim. Scopes are
already canonical strings; no fuzzy matching needed.

### 3.4 Content body shape

`content` is structured prose — not free-form paragraph, not pure JSON.
Borrowed from GraphRAG's report schema but lightened:

```
TITLE: One-line title.

OVERVIEW:
One short paragraph stating the gist. ≤80 words. No bullet points.

FINDINGS:
- [F1] One concrete fact or pattern, citing evidence as [E:<episode_uuid>]
  or [S:<summary_uuid>] inline. ≤25 words each. 5–10 findings max.
- [F2] ...

STAKEHOLDERS / KEY ENTITIES:
- [N:<entity_uuid>] Name — one-line role.

OPEN QUESTIONS:
- Any unresolved or contradicted points the dream noticed. ≤3.

LAST_UPDATED: ISO timestamp.
```

This structure is parseable by regex if we ever need to extract findings
programmatically (e.g., to surface them as bullets in the dream-diff UI of
P6) but renders cleanly in a markdown context. It's strict enough that the
phase-3 sanitizer can deterministically validate format.

Compression target: **0.25–0.35**. That matches RAPTOR's 0.28 reported
ratio. For a subject with 30 KB of contributing episode content, the
summary should be 7-10 KB.

### 3.5 Indexes and constraints

```cypher
CREATE INDEX ON :Summary(uuid)
CREATE INDEX ON :Summary(group_id, scope)
CREATE INDEX ON :Summary(group_id, summary_kind, subject_key)
CREATE INDEX ON :Summary(group_id, status)
CREATE INDEX ON :Summary(group_id, subject_uuid)

-- For fast version-chain walk:
CREATE INDEX ON :Summary(supersedes_uuid)
```

The `(group_id, summary_kind, subject_key)` composite is the critical one
— it's the lookup the dream pass does before deciding ADD vs UPDATE.

There is one constraint we want but FalkorDB does not enforce natively: at
most one `status=active` summary per `(group_id, summary_kind,
subject_key)`. The dream pass enforces this in code: before writing a new
summary, demote the existing active one in the same transaction.

### 3.6 Example queries

**Get the current summary for a project:**
```cypher
MATCH (s:Summary {group_id: $gid,
                  summary_kind: 'scope_summary',
                  scope: 'project:billing-app',
                  status: 'active'})
RETURN s.title, s.content, s.confidence, s.generation, s.created_at
ORDER BY s.generation DESC LIMIT 1
```

**Walk the provenance chain backward from a current summary:**
```cypher
MATCH path = (current:Summary {uuid: $uuid})-[:SUPERSEDES*0..]->(prior:Summary)
RETURN [n IN nodes(path) | {uuid: n.uuid, generation: n.generation,
                            created_at: n.created_at, content_hash: n.content_hash}]
       AS version_history
```

**Find all summaries that referenced an episode (compliance / debug):**
```cypher
MATCH (s:Summary)-[:DERIVED_FROM]->(ep:Episodic {uuid: $episode_uuid})
RETURN s.uuid, s.summary_kind, s.subject_key, s.created_at, s.status
ORDER BY s.created_at DESC
```

**Find all entity_summary chains for active entities in a scope:**
```cypher
MATCH (s:Summary {group_id: $gid, summary_kind: 'entity_summary',
                  scope: $scope, status: 'active'})-[:COVERS]->(e:Entity)
RETURN e.name, s.uuid, s.title, s.confidence, s.generation
ORDER BY s.generation DESC
```

**Get the warm-context summary tier — top-K relevant summaries for a query:**
```cypher
// Hybrid search done via graphiti's client.search_() with summary_config;
// fallback when that's not wired:
MATCH (s:Summary {group_id: $gid, status: 'active'})
WHERE s.scope = $scope OR s.scope = 'real:global'
WITH s, gds.similarity.cosine(s.content_embedding, $query_embedding) AS sim
ORDER BY sim DESC LIMIT 5
RETURN s.title, s.content, sim
```

(FalkorDB has cosine similarity via the `vec.distance` / `vec.cosineSimilarity`
functions depending on version; we'll wrap this in a helper to centralize
the FalkorDB-specific syntax.)

### 3.7 Hierarchy: does a summary summarize summaries?

Three levels are possible:

1. `entity_summary` over episodes for a single entity.
2. `scope_summary` over `entity_summary` nodes + scope-tagged episodes for a scope.
3. `meta_summary` over `scope_summary` nodes — e.g., a `real:global` rollup
   of all the user's projects.

**Decision for v1: ship (1) and (2), defer (3).** Rationale:

- The two levels are conceptually distinct (entity vs. scope) so each has a
  natural subject key. A third level — meta — has to invent its key
  (some hash of scopes) and the natural-language addressing breaks.
- RAPTOR-style automatic recursion (cluster summaries, summarize again)
  *might* be useful for very-heavy users (>20 active scopes), but for the
  expected user shape (1-5 active projects, ≤2 books) the value is small.
- Recursion is where drift compounds most — see §7.5 — so we want to be
  sure phase-1 stability is solid before adding it.

If P12-v2 needs meta-summaries, the schema already accommodates them:
`meta_summary` kind, `source_summary_uuids` populated, `:MERGED_FROM` edges
to each underlying summary. No migration needed.

### 3.8 Garbage collection

The same status enum we have for facts applies to summaries:

| Status | Visible to default search? | Hard-deletable? | Trigger |
|---|---|---|---|
| `active` | yes | no | Just written, latest of its chain. |
| `tentative` | yes | no | Phase 2 output before ratification (rare for summaries — see §8.1). |
| `superseded` | no | yes after retention window | New generation written; old one is the prior in the chain. |
| `contradicted` | no | yes after retention window | A `:CONTRADICTS` edge was added by a later pass. |

**Retention window: 90 days for superseded, 365 days for contradicted.**
Then hard-delete via a background job (`cleanup_old_summaries` in the
scheduler). The shorter window for superseded is fine because the chain is
preserved — you can still walk the version history of an `active` summary
back through `:SUPERSEDES` because we never delete the *latest* superseded
node connected to an active one. We delete *only* superseded nodes that are
themselves the target of a `:SUPERSEDES` edge from another superseded node
(i.e., everything older than the second-most-recent generation), and only
after the retention window expires.

Concretely:

```cypher
// Delete summaries that are at least 90 days old AND are superseded AND
// are not the most recent superseded (i.e., a more-recent superseded
// version exists in the chain).
MATCH (old:Summary {status: 'superseded'})
WHERE old.created_at < datetime() - duration({days: 90})
  AND EXISTS {
    MATCH (newer:Summary {status: 'superseded'})-[:SUPERSEDES]->(old)
  }
DETACH DELETE old
```

This preserves at least one superseded generation per chain forever (as
long as the chain still has an active head), which is enough to walk back
*one* version for the rollback / diff UI in P6 and P7 without storing
unbounded history.

---

## 4. Identity — the hardest problem, walked through

This section walks each scenario the prompt asks about.

### 4.1 The desired invariants

1. **Warm-context retrieval always returns the latest active summary** for a
   given subject. Never returns two summaries about the same subject in
   parallel.
2. **The full provenance chain is always reconstructable** for any summary
   that exists in the database — we can answer "what episodes contributed to
   this claim?" deterministically.
3. **Edits never lose history.** A user-edited summary writes a new version,
   it does not mutate the old.
4. **Re-running a dream pass against the same inputs produces a `NOOP`**
   for unchanged subjects (no spurious new generation).

The fourth is enforced by `content_hash` comparison. The first three are
enforced by the version-chain semantics and the rule that a summary node is
write-once after creation.

### 4.2 Scenario A: dream replaces v3 with v4

Existing state: `Summary(uuid=S3, generation=3, status=active,
subject_key=entity:sarah)`. Tonight's dream sees new episodes about Sarah.

The dream pass's UPDATE flow:

1. Pull the current active summary `S3` for `subject_key=entity:sarah`.
2. Pull all source episodes since `S3.valid_from` plus all episodes already
   referenced by S3 (`S3.source_episode_uuids`).
3. Phase 1 produces a new candidate summary body.
4. Compute `content_hash(new_body)`. If equal to `S3.content_hash`, emit
   `NOOP` and stop. No new node.
5. Otherwise, build `S4`:
   - `uuid` = new UUIDv4
   - `generation` = 4
   - `supersedes_uuid` = `S3.uuid`
   - `status` = `active`
   - `source_episode_uuids` = (S3.sources ∪ new episodes)
   - `source_summary_uuids` = [`S3.uuid`]
6. In one Cypher transaction:
   ```cypher
   // Create S4 node
   CREATE (s4:Summary { ...S4 props... })
   WITH s4
   MATCH (s3:Summary {uuid: $s3_uuid})
   SET s3.status = 'superseded',
       s3.valid_to = datetime(),
       s3.superseded_by_uuid = s4.uuid
   CREATE (s4)-[:SUPERSEDES { group_id: $gid, created_at: datetime() }]->(s3)
   // Re-wire :COVERS edges (entity_summary case)
   WITH s4, s3
   MATCH (s3)-[old:COVERS]->(e:Entity)
   CREATE (s4)-[:COVERS { group_id: $gid, created_at: datetime() }]->(e)
   // (We do not delete old :COVERS — keeping S3's edges intact preserves
   //  historical retrieval. They just don't surface because S3.status != active.)
   ```

The transaction is one Cypher call to guarantee atomicity. If the executor
crashes between writing `S4` and demoting `S3`, both summaries will read as
`active` — which our warm-context query then ranks by generation DESC, so
S4 wins, but we'll have data inconsistency. The dream pass `_commit_summary`
helper does:

- Use Cypher's transactional semantics (`BEGIN`/`COMMIT` or single
  multi-statement query).
- After commit, verify the invariant with a follow-up query: count of
  `active` summaries for this `subject_key` should be exactly 1.
- If not, raise and alert — this is data corruption and should never happen.

### 4.3 Scenario B: user edits v4 manually (P7)

P7 — User veto / edit on dream output (TODO.md). The user opens the dream
session UI, sees `S4`'s content, edits it (let's say corrects "Sarah moved
to billing" to "Sarah is on billing and security review").

The edit flow:

1. Frontend POSTs the new content via a `PATCH /summaries/{uuid}` endpoint.
2. Backend reads `S4`, builds `S5`:
   - `generation` = 5
   - `supersedes_uuid` = `S4.uuid`
   - `status` = `active`
   - `provenance` = `user_edit:{user_id}:{ts}`
   - `source_episode_uuids` = `S4.source_episode_uuids` (unchanged — user
     edit doesn't introduce new sources)
   - `source_summary_uuids` = [`S4.uuid`]
3. Same Cypher commit as Scenario A.
4. The dream pass on the next night sees an `active` summary whose
   provenance is `user_edit:*`. Its UPDATE-decision prompt is biased
   accordingly: "if the latest version of this summary was hand-edited by
   the user, you must explain *why* you're changing it again — i.e., new
   evidence that contradicts the user's edit. Otherwise NOOP."

This is the user-veto signal feeding back into dreaming (P7 ↔ P12 tie-in).
The mechanism is just: `provenance.startswith("user_edit:")` is a flag the
dream-pass prompt checks.

### 4.4 Scenario C: dream regenerates v5 from v4 + new episodes

Standard UPDATE — same as Scenario A, with generation = 5,
supersedes_uuid = S4. No special handling. The version chain now has
S5 → S4 → S3 → S2 → S1, with the gap (S5 → S4) being "user edit followed
by a regenerated dream pass." Walking the chain via `:SUPERSEDES` shows the
full audit trail.

### 4.5 Scenario D: an episode that was input to S3 is later contradicted

This is the trickiest case. Episode E12 was one of S3's evidence sources.
Episode E12 said "Sarah leads the billing team." A later episode E40 says
"Sarah is no longer on billing — she moved to security." The dream pass
phase 1 demotes the *fact* extracted from E12 (sets `status=superseded` on
the `:RELATES_TO` edge) per P0.3.

What happens to S3? Two cases:

**Case D1**: S5 (current active) was written after E40 and already says
"Sarah is on security." Then S3 is already superseded; nothing to do. The
fact change in the underlying edge doesn't propagate up because the chain
moved on.

**Case D2**: The active summary for this subject is *still* S3 (E40 just
arrived, dream hasn't run yet). The fact demotion is independent of summary
status. When tonight's dream runs, it sees:
- Episode E40 (new).
- Existing summary S3, which lists E12 as a source. E12's extracted edge
  has `status=superseded`.

The dream pass's prompt for UPDATE includes a tag on each source episode
indicating whether any of its extracted facts have been demoted since the
last summary generation. This makes the dream-pass aware of "contradiction
events in your sources" and biases toward an UPDATE that explicitly
addresses the contradiction:

```
<sources>
  <episode uuid="E12" status="active" demoted_facts_count="1">
    [E12 content]
  </episode>
  <episode uuid="E40" status="active" demoted_facts_count="0">
    [E40 content]
  </episode>
</sources>
```

Phase 1 then writes S4 with an updated body and (when the contradiction
crosses a confidence threshold) emits a `:CONTRADICTS` edge from S4 to S3.
This is rare and only used when the dream pass wants to flag "this is not
just a refinement, this is a *correction*." Downstream we can use this
edge to surface "the agent learned something it had previously gotten
wrong" in the diff UI (P6).

### 4.6 Two summaries that should be one (identity split)

Failure mode. The dream pass writes two entity summaries with subject_keys
`entity:U1` (Sarah from week 1) and `entity:U2` (Sarah from week 5), because
Graphiti's entity resolution failed and created two `:Entity` nodes for the
same person.

**Detection**: a periodic job (`detect_summary_identity_splits`, runs
weekly) finds pairs of `active` `entity_summary` nodes whose
`content_embedding` cosine similarity exceeds 0.90 *and* whose covered
entities have name-overlap. Sample a small set, present to product /
ops dashboard for review.

**Merge**: the next dream pass takes both summary chains as input, emits a
new `S_merged` with `supersedes_uuid` set to the more recent of the two
heads, and adds an `:SUPERSEDES` edge to the second head too (the schema
allows multiple `:SUPERSEDES` edges per node — the property
`supersedes_uuid` points at the *primary* predecessor but the relationship
captures all). Both old chains are demoted to `superseded`.

This is more complex than the linear chain — the version graph becomes a DAG
rather than a list. The schema supports it; the query "find all ancestors
of this summary" just becomes a graph traversal rather than a linked-list
walk:

```cypher
MATCH (current:Summary {uuid: $uuid})
MATCH path = (current)-[:SUPERSEDES*0..]->(ancestor:Summary)
RETURN DISTINCT ancestor
```

### 4.7 One summary that should be two (identity collapse)

Worse failure mode. The dream pass collapses two distinct subjects into one
summary — e.g., merges "Sarah from work" and "Sarah from book club" into a
single chain. This is harder to detect automatically because by the time
we'd want to detect it, the content already conflates the two.

**Prevention is primary**: the dream pass UPDATE prompt asks "is the
subject of this summary the same individual/concept as the subject we're
seeing in the new episodes?" before merging. If the model isn't confident,
phase 3 splits into two CREATE operations rather than one UPDATE.

**Detection if it happens**: a periodic job
(`detect_summary_identity_collapse`, weekly) for each active summary
re-runs the dream-pass phase-1 prompt on a random 50% sample of its source
episodes and asks "does this subset support the same summary?" — and
flags those where the sampled-subset summary diverges substantially
(cosine ≤ 0.60 vs the original). False positives go to product for
review; confirmed cases trigger a forced split in the next pass.

These are operationally cheap because they run weekly and against a
sample.

---

## 5. Evaluation / metrics

We have P0.6's standing benchmark harness (TODO.md). P12 adds three new
metric families that go into the same harness.

### 5.1 Retrieval-ablation lift

Hold out 50 representative queries per power user — sample of first-turn
waking questions, anonymized via an LLM. Score answer quality on a
1–5 Likert by a `claude-opus-4-7` judge under three retrieval conditions:

| Condition | Warm context contents |
|---|---|
| `facts_only` | Top-N `:RELATES_TO` facts + last 5 episodes (current behavior). |
| `summary_only` | Top-K `:Summary` nodes (no facts, no episodes). |
| `summary_first` | Top-K `:Summary` nodes + top-N facts + last 5 episodes. |

Target: `summary_first` ≥ `facts_only` + 0.3 Likert median lift on the
golden query set, and `summary_only` ≥ `facts_only` − 0.2 Likert (i.e.,
summaries alone shouldn't catastrophically degrade — they should be at
least as good as raw facts for general questions). If `summary_only`
underperforms but `summary_first` wins, we ship the tiered retrieval. If
both underperform, P12 was a waste; we kill it.

This is the killshot metric. It runs in the existing P0.6 harness.

### 5.2 Compression ratio

For each `entity_summary` generated, measure:

```
compression_ratio = len(summary.content) / sum(len(ep.content) for ep in sources)
```

Target: median ratio in `[0.20, 0.40]` across all summary generations,
matching RAPTOR's reported 0.28 ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)).
A ratio below 0.10 means we're over-compressing and probably losing detail;
a ratio above 0.50 means we're not really summarizing.

Cheap to measure — derive from the summary's `source_episode_uuids` and
the body lengths. Plot the histogram in the dream-pass dashboard.

### 5.3 Provenance recall

For each generated `entity_summary`, sample 5 sentences from the FINDINGS
section. For each sentence, prompt a judge LLM with the sentence and the
list of source episode UUIDs + their content, and ask: "which episodes
support this sentence? If none, flag as unsupported."

Targets:
- Per-sentence: ≥90% supported by at least one cited source.
- Per-summary: 0 unsupported findings is the bar. ≥1 unsupported finding
  flags the summary for review.

This is the explicit check against hallucinated findings. Run as a
batched offline eval; sample N=20 summaries per dream pass.

### 5.4 Stability — regeneration semantic equivalence

Pick 30 representative active summaries. Run the dream-pass phase-1 prompt
against the same source episodes twice (with different prompt randomness
or model temperature 0.2 vs 0.0) and compute embedding cosine between the
two regenerations.

Targets:
- Median cosine ≥ 0.95 (regenerated summaries are semantically near-identical).
- ≤5% of pairs below 0.80 cosine (extreme divergence is rare).

Below the targets means our phase-1 prompt is too high-variance and needs
tightening. This catches the "model just decided to phrase it differently
today" failure mode.

We also run an LLM-judge "are these the same summary?" question on the
same 30 pairs as a sanity check — judges sometimes catch semantic shifts
that embedding cosine misses (e.g., negation flipped, attribution wrong).

### 5.5 Cost per consolidation pass

For each dream pass, record:

- Total input tokens to phase 1 (consolidation).
- Total output tokens from phase 1.
- Number of summaries created / updated / NOOPed / retired.
- Wall-clock time.
- $-cost via the model's reported pricing.

We aim for the dream pass to add **at most 30% to baseline (no-summary)
dream cost per user-night**. If users with 5+ active scopes blow past that,
we add a per-user-day budget cap (`dream_summary_budget_usd`) and let the
phase 2 / phase 1 split skip lower-confidence summaries on cost overrun.

This is a guardrail not a target — we expect cost growth to be sub-linear
because content_hash NOOPs short-circuit a lot of the heavy work.

### 5.6 Schedule

Day-zero of P12 ship: P0.6 harness runs all five metric families against a
small cohort (1% of users behind LD flag `dream_summaries`). Weekly review
of dashboards. The retrieval-ablation lift is the gate for expanding
beyond the cohort.

---

## 6. Failure modes — each with a mitigation

The prompt asks for at least seven failure modes. Here are nine, ordered
roughly by severity.

### 6.1 Summary drift — recursive regeneration compounds errors

**The risk**: today's summary is built from yesterday's summary plus a few
new episodes. Each generation introduces a small error or rewording. After
60 generations, the summary diverges substantially from the underlying
episodes. This is the documented LangChain
`ConversationSummaryMemory` failure mode and is the same failure category
as recursive summarization in
[Wu et al.](https://arxiv.org/abs/2109.10862), where the authors note that
summary quality degrades visibly past depth 3 in the tree.

**Mitigations**:

1. **Periodic full rebuilds.** Every 14th dream pass per subject, ignore
   the prior `:Summary` chain and regenerate from raw episodes only. This
   is the "interleaved replay" half of complementary learning systems —
   every so often, the neocortical schema gets reset against the
   hippocampal store. Compare the full-rebuild output to the
   incrementally-updated chain via cosine; alert if divergence > 0.30.
2. **Phase 3 of the dream pass enforces evidence grounding** — every
   finding in the FINDINGS section must cite an `[E:...]` or `[S:...]`. If
   ≥1 finding cites no source, the whole summary is rejected and the
   dream pass falls back to a full rebuild for that subject.
3. **The stability metric (§5.4) is a leading indicator of drift.**
   Regenerations should be semantically near-identical; if they aren't,
   something's wrong with the prompt or the model is making things up.

### 6.2 Lossy provenance

**The risk**: a summary makes a claim. The `source_episode_uuids` list is
incomplete (model forgot to cite, or episode was pruned before write). We
cannot answer "where did this claim come from?" — compliance / debug fails.

**Mitigation**: the phase-3 sanitizer is *deterministic* on this. It
regexes the body for `[E:<uuid>]` / `[S:<uuid>]` references, builds the
set, and *overwrites* `source_episode_uuids` / `source_summary_uuids`
from that set rather than trusting whatever the LLM put in the JSON
output. Findings with no cite are dropped. This makes provenance recall
mechanically guaranteed at write time.

### 6.3 Identity collapse / split

Covered in §4.6 and §4.7. Mitigations: dedup-aware CREATE, weekly
identity-check jobs, conservative phase-1 prompts ("if uncertain whether
the subject is the same as the prior summary's, prefer a separate
chain").

### 6.4 Stale summary blocking fresh facts

**The risk**: warm context retrieves a stale summary that says "Sarah is
on billing" when the user has just told us in this turn that "Sarah moved
to security." Retrieval prefers the summary tier, summary is wrong.

**Mitigations**:

1. **Warm context renders facts and summaries side-by-side** (§8.2). The
   summary is not "instead of" facts — the LLM reading the system prompt
   sees both, and Graphiti's fact tier reflects the most recent edges
   (including just-extracted ones from the in-flight session ingest).
2. **Summaries have a `valid_from` timestamp visible to the chat
   prompt.** If the summary is more than 14 days old and the user's
   current turn introduces new entities related to the subject, the chat
   model is instructed to weight the user's own words over the summary's
   claims.
3. **The dream pass that writes a summary records the `instruction_hash`
   of the dream-pass instructions used to write it.** If the user
   significantly changes their dream instructions, future warm contexts
   can flag "this summary was written under different instructions"
   (low-priority — defer to v2).

### 6.5 Recursion bombs (only for meta_summary, v2)

If meta-summaries ever ship, the risk is: meta_summary → scope_summary →
entity_summary → episode, and meta_summary's prompt accidentally embeds
the full transitive content. Total prompt size explodes.

Mitigation when shipping v2: hard cap on `meta_summary` content body at
3,000 chars (deterministic enforcement in phase 3) regardless of input
size. If a meta_summary needs more, split into multiple meta_summaries
keyed on sub-domain. Also: meta_summary prompt only includes the TITLE +
OVERVIEW of source summaries, never their full FINDINGS sections.

### 6.6 Summarizing inconsistent inputs without flagging the inconsistency

**The risk**: two source episodes flatly contradict each other ("Sarah is
on billing" / "Sarah is on security"). The summary picks one without
acknowledging the conflict; the user thinks the agent missed the update.

**Mitigation**: the phase-1 prompt explicitly requires the OPEN QUESTIONS
section to list contradictions found across sources, with both citation
sets. Phase-3 verifies that any pair of source episodes whose extracted
facts about the same entity disagree (detectable via the existing fact
embeddings and the contradiction-detection logic from P0.3) is either
resolved in the FINDINGS section or noted in OPEN QUESTIONS. Failure to
do either → reject summary.

### 6.7 Schema overhead degrades chat latency

**The risk**: warm-context now does (a) facts, (b) episodes, (c)
summaries. Three queries instead of two. Latency budget is 5 seconds for
warm context (`graphiti_config.context_timeout`). We're now 50% closer to
that ceiling.

**Mitigations**:

1. The three queries run *in parallel* via `asyncio.gather` — the
   bottleneck is the slowest, not the sum. Summary query is cheap (one
   indexed lookup + cosine on ~10 candidates) so we expect it to be
   under-3-second p95.
2. Hard cap: `summary_query_timeout = 1.5s`. On timeout, the summary tier
   silently drops and warm-context falls back to facts+episodes alone.
   This is the same graceful-degradation pattern context.py:38-46
   already uses for the whole warm context.
3. Latency goes in the benchmark harness; flag the run if p95 > 4s.

### 6.8 Embedding drift across embedder versions

**The risk**: today's `content_embedding` was made with
`text-embedding-3-small`. We upgrade the embedder. Summary cosine
comparisons (stability, identity-split detection) now compare across
embedding spaces and produce nonsense.

**Mitigation**: store `embedding_model` and `embedding_model_version` as
properties on `:Summary`. The stability eval and the identity-check
jobs both filter to summaries embedded with the same model+version. On
embedder upgrade, run a background job to re-embed active summaries
(`backfill_summary_embeddings`) before relying on cosine comparisons.

### 6.9 The dream pass writes a summary, then the user invokes "forget" on the
covered entity

**The risk**: user uses `MemoryForgetConfirmTool` to soft-delete some
edges related to Sarah. Tomorrow's dream pass still has Sarah's summary
chain, which now contains claims contradicted by the forget action.

**Mitigation**: the forget tool's `_soft_delete_edges` helper gets
extended to also mark any `entity_summary` whose `:COVERS` edge points at
the affected entity as `status=tentative` with a `provenance` suffix of
`forget_pending_recompute`. The next dream pass treats these as needing
recompute regardless of `content_hash` (it forces a regeneration to get
a clean summary post-forget). After the next dream, the summary is
either `active` again (if the recompute produced clean content) or
remains `tentative` and skips warm context.

---

## 7. Build plan — concrete files, in order, with engineer-week estimates

**Total estimate: 3.0 engineer-weeks for v1 (P12a + P12b)**, with P12c
(meta-summaries, cross-scope summaries, dedicated retrieval API) as a v2
~1.5-week extension landing after metrics validate v1.

### P12a — Schema + primitives (≈1.0 engineer-week)

These are independent of the dream pass and can be built and tested in
isolation against an empty user. They are not user-visible until P12b
wires them in.

| # | File | Action | Notes |
|---|---|---|---|
| 1 | `backend/copilot/graphiti/summary_model.py` | NEW | Pydantic `SummaryEnvelope` (mirrors `MemoryEnvelope` shape but with summary-specific fields). Add `SummaryKind` enum (`entity_summary`, `scope_summary`). |
| 2 | `backend/copilot/graphiti/summary_ddl.py` | NEW | Idempotent index/constraint creation, called from the same boot path that ensures FalkorDB schema. Mirrors existing graphiti ensure_indices_and_constraints flow. |
| 3 | `backend/copilot/graphiti/summary_repo.py` | NEW | `create_summary(envelope) -> uuid`, `get_active_summary(group_id, kind, subject_key) -> SummaryRecord \| None`, `get_chain(uuid) -> list[SummaryRecord]`, `find_by_entity_uuid(group_id, entity_uuid) -> list[SummaryRecord]`. All raw Cypher, no graphiti-core methods. |
| 4 | `backend/copilot/graphiti/summary_commit.py` | NEW | `commit_summary_update(envelope, supersedes_uuid)` — the atomic Cypher transaction from §4.2. Handles UPDATE/CREATE/NOOP/RETIRE verbs. |
| 5 | `backend/copilot/graphiti/summary_search.py` | NEW | `top_k_summaries(group_id, query_embedding, scope, k=5)` — the warm-context retrieval. |
| 6 | `backend/copilot/graphiti/summary_repo_test.py`, `summary_commit_test.py`, `summary_search_test.py` | NEW | Unit tests against the existing FalkorDB test infra used by `ingest_test.py`. |
| 7 | `backend/copilot/tools/graphiti_forget.py` | EDIT | Extend `_soft_delete_edges` to also flip covering summaries to `tentative` per §6.9. |

**Parallelizable with P0 work?** Mostly yes. The schema work doesn't
touch anything the P0 dream pass reads or writes. The only risky overlap
is the `graphiti_forget.py` edit, which is small and adjacent to existing
code.

### P12b — Wire into dream pass + warm context (≈1.5 engineer-weeks)

Depends on P12a and on P0 (the dream pass framework itself).

| # | File | Action | Notes |
|---|---|---|---|
| 1 | `backend/copilot/executor/dream/phases.py` (NEW under P0) | EDIT | Phase 1 now produces a `proposed_summaries` list alongside `consolidated_facts` and `demotions`. Phase 2 unchanged. Phase 3 validates summary structure and runs the ADD/UPDATE/NOOP/RETIRE decision. |
| 2 | `backend/copilot/executor/dream/summary_decision.py` | NEW | The Mem0-style decision LLM call. For each candidate summary, looks up the active chain by `subject_key`, returns one of `CREATE`, `UPDATE`, `NOOP`, `RETIRE` plus the new body. |
| 3 | `backend/copilot/executor/dream/prompts.py` (NEW under P0) | EDIT | Add summary-specific prompts: subject-key derivation, structured-body generation, evidence-citation enforcement. |
| 4 | `backend/copilot/graphiti/context.py` | EDIT | `_fetch` now does three-way `asyncio.gather`: search edges, retrieve episodes, *and* `top_k_summaries`. Add `<SUMMARIES>` section to `_format_context`. |
| 5 | `backend/copilot/graphiti/context_test.py` | EDIT | Snapshot test the new three-tier warm context format. |
| 6 | `backend/copilot/prompting.py` | EDIT | Update the system-prompt assembly so the LLM understands the new `<SUMMARIES>` block is "long-term consolidated knowledge" vs `<FACTS>` "current relationship facts" vs `<RECENT_EPISODES>` "most recent raw conversations." |

The dream-pass integration is the biggest single chunk of work — it's
not a trivial wire-up because the phase-1 prompt needs to be augmented
with the prior summary chain for each candidate subject, which requires
candidate-subject extraction *before* the main consolidation runs. We do
that with a cheap LLM precheck (Sonnet, temp 0) that looks at the input
episodes + active facts and returns a list of subject keys to consider.
Then for each subject, the main phase-1 pass fetches the active summary
and decides UPDATE/CREATE/NOOP.

### P12c — Meta-summaries and cross-scope (≈1.0 engineer-week, v2 only)

Land only if §5.1 lift > +0.5 Likert in v1.

| # | File | Action | Notes |
|---|---|---|---|
| 1 | `summary_model.py` | EDIT | Add `meta_summary` to `SummaryKind`. |
| 2 | `summary_meta.py` | NEW | The meta-summary build pass — input is the set of `scope_summary` heads, output is one or more `meta_summary` nodes. |
| 3 | `dream/cross_scope_pass.py` | NEW | The cross-scope discovery pass (P8 tie-in) that consumes `meta_summary` to look for transferable patterns. |
| 4 | `summary_search.py` | EDIT | Add `meta_summary`-aware retrieval that boosts top-level summaries for ambiguous queries. |

### Frontend (deferred to P6 / P7 surfacing)

P12 itself ships with **no UI** beyond an admin debug page showing summary
counts and recent generations. User-facing surfacing of summaries (and
the user veto of §4.3) happens via P6 + P7, which already plan to expose
dream output. The summary tier is a natural enrichment of those flows
once they exist.

### Cutover plan

1. P12a lands behind no flag (schema is dormant; no writes, no reads).
2. P12b ships behind LaunchDarkly flag `dream_summaries`. Default off.
3. Roll to 1% of users; gate expansion on §5 metrics for two weeks.
4. Expand to 10% / 50% / 100% based on metrics. If §5.1 lift is < 0.0
   at 1%, kill the feature and roll back the schema (data stays — no
   migration needed because it was never read by general users).

---

## 8. Tie-in with the rest of the dream roadmap

### 8.1 P0.4 ratification loop

Question from the prompt: do summaries get ratified, or just their
constituent facts?

**Decision**: summaries do *not* enter the ratification loop. Concretely:

- The dream pass writes `entity_summary` and `scope_summary` with
  `status=active` directly. Phase 3's deterministic validation is enough
  — there's no concept of a "tentative summary" for the chat pass to
  retrieve and ratify.
- The facts and findings that are *citations within* a summary still
  follow the standard `tentative` → `active` ratification path. The
  summary references them by UUID; if a cited fact is later demoted,
  §4.5 / §6.9 cover the cleanup.

Why this asymmetry: a summary is by construction *the* summary for a
subject. There's no "trial summary" the chat can decide to use or not.
Either it's the current view of the subject or it isn't. The ratification
machinery is for individual claims, which is the right granularity for
warm-context filtering.

Exception: if phase 3 validation passes with weak confidence (model
self-reports `confidence` < 0.5), the summary is written
`status=tentative` and excluded from default warm context until the next
dream pass either upgrades it to `active` or drops it. This is rare and
mostly for the cold-start case where there isn't enough source signal.

### 8.2 P2 dedup

Summaries supersede dedup for the *subject identity* layer — instead of
asking "are these two facts duplicates", the summary pass asks "what's the
canonical view of this subject right now." Dedup still matters for the
fact tier (we don't want 14 nearly-identical edges about Sarah).

**Composition**: P2's dedup pass runs *before* the summary phase in each
dream. Dedup collapses near-duplicate edges, summary phase ingests the
deduplicated set. This is the natural order: summarize a clean fact base,
not a noisy one.

### 8.3 P5 goals-aware dreaming

P5 biases the dream pass to prioritize user goals. For P12, that means
the dream pass's subject-key precheck (which subjects get summary
attention this run) is biased by goal-relevance:

- User has open agents tagged with `project:billing` → boost
  `scope:project:billing` and any entities tagged in that scope.
- User has a stated plan ("I'm shipping the redesign next Friday") → boost
  the entities and scope mentioned in the plan.

This is one line in the phase-precheck prompt: "prioritize subjects
that intersect with the user's active goals: `<goals_block>`." The
schema doesn't change.

### 8.4 P7 user veto / edit on summary

Covered in §4.3 — `provenance=user_edit:*` is the signal, the dream-pass
prompt respects it. The frontend `/summaries/{uuid}` PATCH endpoint
writes the new version through the same `commit_summary_update`
primitive used by the dream pass — there is *no* second code path. This
keeps the version-chain invariants intact.

User veto is also the test of how seriously we take the immutability
contract. If a user edits a summary and the next dream pass overwrites
their edit, the user will (correctly) feel like their input doesn't
matter. The "user-edit gate" in the dream-pass prompt is the explicit
respect-the-user mechanism.

### 8.5 P1 procedure synthesis

Procedures and summaries are *orthogonal*, not nested. A
`ProcedureMemory` is a structured artifact for "how to do X" — discrete
steps, conditions, negations, suitable for re-execution. A `:Summary`
node is descriptive text about a subject — "what we know about X."

They tie in via cross-reference: an `entity_summary` can mention "the
user has a saved procedure for closing out monthly invoices, see
proc:abc". The summary's FINDINGS section can cite procedures by UUID
the same way it cites episodes. But procedures are first-class objects
(per TODO.md P1), and summaries don't summarize procedures.

### 8.6 P8 cross-scope discovery

Cross-scope discovery is the meta-summary use case. Once `scope_summary`
nodes exist per scope, the cross-scope pass takes the set of active
`scope_summary` nodes for a user and looks for transferable patterns
("the workflow from `project:billing` would apply to `project:auth`").
The output is either:

- A new `meta_summary` node (v2) with `:MERGED_FROM` edges to the source
  scope_summaries, or
- A `MemoryEnvelope` of kind `finding` written via the normal
  enqueue_episode path (v1), with `scope=real:global` and provenance
  `dream:cross_scope`.

Either way, the schema accommodates it without further design work.

---

## 9. Compare against alternatives

Required by the prompt: three-way comparison of A vs B vs C.

### A. Use Graphiti `:Community` as-is

| Aspect | Notes |
|---|---|
| Integration cost | Lowest. Call `graphiti.build_communities(group_ids=[gid])` from the dream pass. Communities surface via `client.search_()` with `community_config`. |
| Pros | Zero new code in our codebase. Upstream-maintained. Falls into graphiti's existing search pipelines. |
| Cons | Full rebuild every pass (no stable identity). No typed metadata (`status`, `confidence`, `provenance`, `supersedes_uuid`). Cluster-based — does not align with our scope taxonomy. No way to attach to a single entity / scope. Cannot tell the dream pass "summarize Sarah" — communities are auto-discovered, not user-addressable. |
| Exit cost if we change our mind | Trivial — communities can be dropped by deleting nodes. But we'd also lose all the work invested in the prompt + clustering tuning we'd need to do to get them to be useful. |
| Suitability | Wrong abstraction. The cluster boundaries do not match either our scope taxonomy or the per-subject summary the warm-context layer wants. |

**Recommendation: reject for v1.** Consider exposing communities as an
*additional* retrieval signal much later (v3+), only if richer
graph-clustering retrieval becomes a need we don't have today.

### B. Custom `:Summary` node type — what this doc proposes

| Aspect | Notes |
|---|---|
| Integration cost | Highest of the three. Three new files for schema + repo, one new dream-pass phase, warm-context wiring. ~3 engineer-weeks v1. |
| Pros | Stable identity across regenerations; full provenance; addressable by subject key; integrates cleanly with existing scope and `MemoryEnvelope` patterns; future-proof (meta-summaries, cross-scope, user-edit all fit the same schema). The retrieval pattern (summary tier over fact tier over episode tier) is the proven shape (LlamaIndex DocumentSummaryIndex, RAPTOR, GraphRAG). |
| Cons | We own the code. We own the prompt tuning. We own the metrics harness. We own the migration if FalkorDB / Graphiti evolves and the `:Summary` label collides with something upstream introduces. |
| Exit cost if we change our mind | Moderate. The `:Summary` nodes are in our database; we can stop reading them (warm-context flag off) and clean them up with a one-off Cypher in the worst case. The dream pass still writes facts via the existing `enqueue_episode` path, which would continue working. |
| Suitability | Matches the requirements. Worth the build cost. |

**Recommendation: ship.**

### C. Move summaries to Postgres alongside `ChatMessage`

| Aspect | Notes |
|---|---|
| Integration cost | Medium. New Prisma model `MemorySummary`, repository, migrations. Bypasses Graphiti entirely for the summary tier. |
| Pros | Easier to query relationally; clean foreign keys to `ChatMessage` for provenance to specific messages; transactional with the rest of the user's Postgres data; familiar tooling. Faster point-lookup by primary key. |
| Cons | Loses the graph join with `:Entity` and `:Episodic`. Cannot answer "which summaries cover entity X" without dual-querying both stores. Warm context now has to fan out to Postgres + Graphiti — adds a join in the hot path. Embeddings need a parallel index (pgvector). Cross-system writes have no shared transaction guarantee — a half-failed dream pass leaves Postgres summaries pointing at non-existent Graphiti episodes. |
| Exit cost if we change our mind | High. Postgres migrations are forever; tables stay. Application code would need to be rewritten to read from FalkorDB if we wanted to move summaries to the graph later. |
| Suitability | The relational case is real, but the cross-store split is a big architectural commitment that decouples the summary lifecycle from the rest of memory. |

**Recommendation: reject.** The cross-store dual-write is the worst kind
of complexity. The minor wins (point-lookup, relational FKs) don't pay
for it.

### Side-by-side decision matrix

| Criterion | A. `:Community` | B. `:Summary` (proposed) | C. Postgres |
|---|---|---|---|
| Stable identity across regen | No | Yes | Yes |
| Provenance to episodes | Indirect (via members) | Direct (`:DERIVED_FROM`) | Cross-store join |
| Hierarchy / scope alignment | Cluster-based, doesn't match scopes | Native scope + entity keys | Native via FKs |
| Retrieval integration cost | Zero (already in search pipeline) | Moderate (new tier) | High (fan-out across stores) |
| User-addressable subjects | No | Yes (`subject_key`) | Yes |
| Exit cost | Trivial | Moderate | High |
| Eng-weeks v1 | ~0.5 | ~3 | ~2.5 |
| Best for our case | No | **Yes** | No |

---

## 10. Open product / engineering questions

Most decisions in this doc are made. These ones are still open and need
product/PM input:

1. **Default scope for `entity_summary`.** When an entity is mentioned in
   episodes across multiple scopes, which scope owns the entity summary?
   *Recommend*: the scope where the entity was first introduced, fallback
   to `real:global` if cross-scope. Bias subject_key to that.
2. **Retention windows.** `superseded` = 90 days, `contradicted` = 365
   days. Are these compliance-driven (legal hold for org accounts)? PM
   to confirm — these are easily changeable in the cleanup job.
3. **User-visible name for the summary tier.** "Schemas"? "Summaries"?
   "What I know about [subject]"? The internal name `:Summary` is fine;
   the UI surface needs a product call.
4. **Embedding budget.** `text-embedding-3-small` per summary write: each
   `:Summary` node embeds title+content (~10 KB), so ~$0.0002 per write.
   At 5 summaries/user/night × 10k users × 365 days = ~$3.6k/year. Trivial.
   Re-embedding on model upgrade is a one-shot job.
5. **Privacy: do summaries cross organizational boundaries when we add
   orgs?** Per the deferred-items list in TODO.md, cross-user community
   memory is privacy-sensitive but becomes natural at the org level.
   For P12: summaries are strictly per-user. Org-level summaries are a
   separate design problem that we'll tackle when org-scoping lands.
6. **What does a summary chain look like in the dream-diff UI (P6)?**
   Showing "v3 → v4: 2 findings added, 1 retired, OVERVIEW reworded" is
   doable from the diff of the two content bodies plus the citation
   sets. P6 design owns this.

---

## 11. Summary in one paragraph

We add a `:Summary` node label to FalkorDB. The dream pass writes one
node per subject (entity or scope) per generation it changes, never
overwriting the prior version — the version chain is preserved via
`:SUPERSEDES` edges and a `supersedes_uuid` property. Each summary
carries structured prose (title, overview, findings with inline source
citations, stakeholders, open questions), an embedding, full episode-
and summary-level provenance, and the standard `MemoryStatus` ladder.
Warm-context retrieval renders the latest active summary as a new
`<SUMMARIES>` block above the existing `<FACTS>` and `<RECENT_EPISODES>`.
The dream pass uses Mem0-style ADD/UPDATE/NOOP/RETIRE verbs to decide
what to do per subject. Identity is stable because the subject key is
deterministic; drift is bounded because every 14th regeneration does a
full rebuild from raw episodes; provenance is guaranteed by phase-3's
deterministic citation-extraction. The build is ~3 engineer-weeks across
two phases (P12a primitives, P12b dream integration), behind a LD flag,
gated on a five-metric harness extending P0.6. Ship to 1% on schema
land; expand on retrieval-lift evidence.

---

## Sources

Primary references cited in the body, grouped by topic.

**Anthropic Dreams**
- [Dreams docs](https://platform.claude.com/docs/en/managed-agents/dreams)
- [Memory stores docs](https://platform.claude.com/docs/en/managed-agents/memory)

**Graphiti / Zep**
- [Zep paper (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- Upstream graphiti-core community ops: `graphiti_core/utils/maintenance/community_operations.py` in the installed package (lines 217-258 for `build_communities` and `remove_communities`).

**Generative Agents / MemoryBank / Sleep-time**
- [Park et al. 2023 (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442)
- [Park et al. ar5iv](https://ar5iv.labs.arxiv.org/html/2304.03442)
- [Zhong et al. MemoryBank (arXiv:2305.10250)](https://arxiv.org/abs/2305.10250)
- [Letta sleep-time blog](https://www.letta.com/blog/sleep-time-compute)
- [Letta agent memory blog](https://www.letta.com/blog/agent-memory)

**MemGPT / Letta**
- [MemGPT paper (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)
- [Letta MemGPT concepts](https://docs.letta.com/concepts/memgpt/)
- [Letta memory management](https://docs.letta.com/advanced/memory-management/)
- [Mem0 paper (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)

**Recursive Summarization / Hierarchical**
- [Wu et al. OpenAI recursive book summarization (arXiv:2109.10862)](https://arxiv.org/abs/2109.10862)
- [RAPTOR (arXiv:2401.18059)](https://arxiv.org/abs/2401.18059)
- [RAPTOR HTML](https://arxiv.org/html/2401.18059v1)
- [LangChain memory migration overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LlamaIndex Tree Index](https://developers.llamaindex.ai/python/framework-api-reference/indices/tree/)
- [LlamaIndex Summary Index](https://developers.llamaindex.ai/python/framework-api-reference/indices/summary/)
- [LlamaIndex Document Summary Index](https://developers.llamaindex.ai/python/framework-api-reference/indices/document_summary/)

**Graph-native RAG**
- [HippoRAG (arXiv:2405.14831)](https://arxiv.org/abs/2405.14831)
- [HippoRAG HTML](https://arxiv.org/html/2405.14831v1)
- [Microsoft GraphRAG (arXiv:2404.16130)](https://arxiv.org/html/2404.16130v2)
- [Microsoft GraphRAG docs](https://microsoft.github.io/graphrag/)

**Cognitive science**
- [McClelland, McNaughton, O'Reilly 1995 (PDF)](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)
- [Tse et al. 2007 schemas and consolidation](https://pubmed.ncbi.nlm.nih.gov/17412951/)
- Bartlett, *Remembering* (1932) — schema theory in reconstructive memory.

**Stability / Drift / Evals**
- [SSGM framework — governing evolving memory in LLM agents](https://arxiv.org/html/2603.11768v1)
- [MemoryAgentBench (arXiv:2507.05257)](https://arxiv.org/abs/2507.05257)

**Existing platform docs referenced**
- `dream/dreaming-research.md` — full system inventory; Parts I-IV.
- `dream/dreaming-memory.md` — Graphiti client surface, FalkorDB layout, "What's Missing for Dreaming" priority list.
- `dream/dreaming-spec.md` — three-phase dream-pass anatomy, prompt drafts, eval/failure taxonomy.
- `dream/dreaming-anthropic.md` — close read of Anthropic Dreams API.
- `dream/dreaming-chat.md` — executor turn flow and SSE event surface.
- `dream/TODO.md` — full roadmap, this doc covers P12.

In-repo file references used to validate the build plan:
- `autogpt_platform/backend/backend/copilot/graphiti/client.py` — `derive_group_id`, `get_graphiti_client`, per-loop cache.
- `autogpt_platform/backend/backend/copilot/graphiti/ingest.py` — `enqueue_episode`, per-user serialization, `CUSTOM_EXTRACTION_INSTRUCTIONS`.
- `autogpt_platform/backend/backend/copilot/graphiti/context.py` — current two-tier warm context (facts + episodes); the target wiring site for the new summary tier.
- `autogpt_platform/backend/backend/copilot/graphiti/memory_model.py` — `MemoryEnvelope` / `SourceKind` / `MemoryKind` / `MemoryStatus` enums to mirror in `SummaryEnvelope`.
- `autogpt_platform/backend/backend/copilot/graphiti/_format.py` — attribute-extraction helpers; extend with `extract_summary_*` peers.
- `autogpt_platform/backend/backend/copilot/tools/graphiti_forget.py` — `_soft_delete_edges` pattern to mirror, plus the §6.9 edit target.
- `autogpt_platform/backend/backend/copilot/tools/graphiti_search.py` — current scope-filter pattern; extend or mirror for summary search.
- Installed `graphiti_core/utils/maintenance/community_operations.py` — reference implementation of pairwise summarization we explicitly reject in favor of phase-aware structured-body generation.
