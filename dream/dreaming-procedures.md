# Dreaming — Procedure Synthesis and Save-as-Skill (P1)

A design + build plan for the day-two scope of the dream system: detecting repeated workflows in chat history, persisting them as structured procedures, and shipping a user-facing **Save as Skill** affordance that lets users (and the dream pass itself) promote a procedure into a durable, retrievable, optionally shareable artifact.

This is a research-and-design document. It does not write code. Its purpose is to (a) survey the prior art so we know what to imitate and what to avoid, (b) verify the assumptions baked into `dream/TODO.md` against the actual codebase, (c) make a defensible recommendation on the first-class-object question the product owner asked, and (d) hand the eventual implementer a concrete file-by-file plan.

Cited claims include URLs; uncertain claims are labeled as speculation.

---

## 0. Tl;dr for the product owner

Three recommendations, defended throughout the document and summarized here so the reader has the conclusions before the evidence.

1. **A "Skill" is its own first-class object, but it is a thin Postgres row over an existing artifact, not a parallel runtime.** The Postgres row binds together (a) an authored or auto-extracted `SKILL.md`-style document, (b) optional metadata (parameters, trigger conditions, version, visibility), and (c) a pointer to one of three substrates: a `ProcedureMemory` envelope in Graphiti, an existing `AgentGraph` template in the library, or — for the simplest skills — nothing but the markdown body itself. A skill is best understood as **a curated, named pointer to behavior**, exactly like Anthropic's `SKILL.md` model. We get the discoverability and sharing benefits of a first-class object without inventing a new execution runtime. ([Anthropic Skills overview](https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/overview))

2. **Save as Skill is a real tool, not just a UI button.** It is invokable by the user in chat ("save that as a skill"), by the dream pass when it detects a high-confidence repeated procedure, and via slash command/library UI. The tool's signature, surfaces, server-side behavior, and lifecycle (`draft` → `tentative` → `active` → `superseded`) are spelled out in §5.

3. **Pattern detection is a hybrid pipeline**: a cheap heuristic surfaces candidates from chat sessions and tool-call sequences (process-mining-style alpha/inductive detection over the directly-follows graph of tool calls + repeated user-message embedding clusters), then an LLM judge confirms and structures them. This runs inside Phase 2 of the dream pass and emits `proposed_finding` / `proposed_procedure` envelopes that go through the normal ratification loop.

Everything else in this document explains why those three choices fit our codebase, what the alternatives are, and what to build first.

---

## 1. Prior art — what to steal and what to avoid

### 1.1 Anthropic Claude Skills (the closest model)

**What it is.** A Skill is a filesystem directory containing a `SKILL.md` with YAML frontmatter (`name`, `description`, optional `license`, `metadata`), an optional markdown body of procedural instructions, optional sub-reference files (`FORMS.md`, `REFERENCE.md`, etc.), and optional executable scripts. Claude loads the metadata of every available skill at startup, the body of a skill only when the description matches the current task, and bundled files only when the skill body references them. This is the **progressive-disclosure pattern** Anthropic ships in `~/.claude/skills/`, in `claude.ai` Settings → Features, in Claude Code projects (`.claude/skills/`), and via the `/v1/skills` API endpoints. ([Anthropic Skills overview](https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/overview), [Anthropic Skills launch post](https://claude.com/blog/skills), [Best practices](https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/best-practices))

**What's interesting.**
- Two required frontmatter fields and that's it: `name` (≤64 chars, lowercase + hyphens, no `anthropic`/`claude` reserved words) and `description` (≤1024 chars). Everything else is optional. The description is load-bearing — it's the only thing in context until the skill fires.
- "Three levels of loading" — metadata (always), body (on trigger), bundled files (on reference). Claude reads files using bash; non-loaded files cost zero tokens.
- Sharing model is **per-surface**: workspace-scoped on the API, individual-user on `claude.ai`, filesystem-only in Claude Code. There is **no cross-surface sync**. Users that want a skill on multiple surfaces must upload it to each.
- Authoring flow is iterative and pair-Claude'd: "Claude A" (helper) writes the skill, "Claude B" (consumer) uses it; the user observes Claude B's failures and feeds them back to Claude A.
- An emerging open standard at [agentskills.io](https://agentskills.io). Whether that becomes universal is unknown — speculation: we should not assume it, but our shape should be **compatible-by-projection** with it (described below).

**What to steal.**
- **The SKILL.md format itself.** `name` + `description` frontmatter + markdown body is the right shape. It is portable, human-readable, and already a recognizable artifact for the user base most likely to power-use our dream feature.
- **Progressive disclosure.** Cheap to keep many skills available; only load the ones that fit. Our equivalent is: skill metadata in the warm-context block, skill body fetched only when the LLM decides to invoke it.
- **The "Claude A / Claude B" authoring loop.** Maps cleanly onto the dream pass (Claude A) proposing a skill from a user's transcripts, the user accepting/editing, and the chat agent (Claude B) using it next session.
- **The conciseness gospel.** Every token in a skill costs context. Our prompts for procedure synthesis should explicitly aim for "minimal-information-additional" — skip what the model already knows.

**What to avoid.**
- **No first-class sandbox.** Anthropic's skills assume a code-execution VM with bash. We have a tool registry instead (§2). Trying to bolt a per-skill VM onto our copilot is out of scope for v1.
- **Anthropic's flat-store-of-text design.** Our `MemoryEnvelope` is richer (typed status, confidence, scope). We should not flatten it to keep parity with Anthropic; we should keep our typing and project to a `SKILL.md`-compatible export only when the user explicitly exports or shares.
- **Per-surface sharing fragmentation.** We have a single product; we should not replicate Anthropic's fragmentation. One skill, accessible to the chat agent, to scheduled agent graphs, and to the marketplace, with the same source-of-truth row.

### 1.2 Voyager (Wang et al., 2023) — the canonical academic precedent

**What it is.** An LLM-driven embodied agent in Minecraft that builds an ever-growing skill library of JavaScript functions. Skills are stored indexed by an embedding of their natural-language description; when a new task appears, the top-5 most relevant skills are retrieved and injected into the prompt. New skills are added when the agent successfully completes a task that the existing library could not solve, validated by GPT-4 acting as a critic. ([Voyager project page](https://voyager.minedojo.org/), [arXiv 2305.16291](https://arxiv.org/abs/2305.16291))

**What to steal.**
- **Embed the description, search the description.** Voyager's retrieval is description-embedding similarity, not code or steps. This is exactly how Anthropic Skills work (metadata in context, body on trigger), and it maps onto our existing Graphiti vector store with no new infra.
- **Skills as compositional code.** New skills can call existing skills. In our world that means: a saved skill can call other saved skills (`run_skill(...)` inside a procedure body). Compounding lifts the ceiling.
- **Self-verification by LLM critic before adding.** Don't write a skill until the LLM judges it would have succeeded. Maps onto our Phase 3 "sanitize and commit" gate in `dreaming-spec.md` §1.
- **Skill descriptions need to be discriminative.** Voyager bakes this in via prompt examples; we should bake it in via the authoring tool's description-quality check.

**What to avoid.**
- Voyager skills are executable code in an environment with a deterministic simulator. Ours run against a stateful platform with side effects on real user resources. We need a stricter gate before any skill auto-executes (vs. read-only "advice" skills). See §5.6 and §8.4.

### 1.3 OpenAI Custom GPTs and Actions

**What it is.** A Custom GPT is a packaged prompt + uploaded knowledge files (≤20, ≤512 MB each) + optional Actions (third-party API integrations specified via OpenAPI). Authoring is via the GPT Builder (conversational) or Configure tab. Sharing is per-GPT: private, link-shared, public via the GPT Store, or workspace-internal. ([OpenAI: Creating and editing GPTs](https://help.openai.com/en/articles/8554397-creating-and-editing-gpts), [OpenAI: GPTs overview](https://openai.com/index/introducing-gpts/))

**What to steal.**
- **The conversational authoring builder.** OpenAI's GPT Builder lets the user iterate on a GPT by chatting with it. Our analogue is: the user is talking to the chat agent, asks "let's clean this up and save as a skill", and the Save-as-Skill tool walks them through name/description/parameters in the same conversation. Don't make them navigate to a separate builder UI for the first version.
- **Knowledge files** as a substrate. Our equivalent is workspace files referenced by the skill body — we already have `WorkspaceManager` and `read_workspace_file`. A skill can ship with workspace artifacts.

**What to avoid.**
- **The actions/OpenAPI plumbing.** That's blocks/MCP territory in our world; do not invent a third path. Skill behavior is expressed as instructions that call our existing tools.
- **The marketplace-first framing.** Most GPTs in the store are duds and the store has notable spam problems. We should make sharing explicit and ratification-gated, not a default behavior.

### 1.4 LangChain / LangGraph and "Deep Agents" SKILL.md

LangChain shipped "LangChain Skills" in 2026 explicitly modeled on Anthropic's `SKILL.md` for their Deep Agents harness. Skills are dynamically loaded via progressive disclosure; the agent only loads a skill when relevant. ([LangChain Skills](https://www.langchain.com/blog/langchain-skills), [docs.langchain.com Skills](https://docs.langchain.com/oss/python/deepagents/skills))

**Implication.** The `SKILL.md` shape is becoming the de-facto industry standard. Adopting it (or a superset of it) gives us interop benefits we don't have to design for explicitly. **We should make our skill format readable as `SKILL.md`** so a power user can copy-paste between Claude Code and our platform.

### 1.5 LangMem — procedural memory in LangChain

LangMem ships **procedural memory** as one of three first-class memory types (semantic, episodic, procedural). Procedural memories are "updated instructions in the agent's prompt" — the optimizer extracts behavioral patterns from successful/unsuccessful interactions and rewrites the system prompt to reinforce them. ([LangMem SDK launch](https://blog.langchain.com/langmem-sdk-launch/), [LangMem concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/))

**What to steal.**
- The framing — **procedural memory = behavior rules the agent applies to itself** — is the right mental model. Maps onto our `MemoryKind.procedure` + `ProcedureMemory` envelope directly.
- The feedback loop — procedures evolve based on observed outcomes, not just one-shot extraction. We get this for free from the dream pass's ratification mechanism (`tentative` → `active` once warm-context retrieval or user reference confirms).

**What to avoid.**
- **Self-rewriting system prompts.** LangMem will modify the system prompt of the running agent. Our system prompt is static and Langfuse-managed (`dreaming-chat.md` §4) precisely because it's cache-keyed. A procedure should be **content the agent reads**, not **a mutation of the system prompt**.

### 1.6 MemGPT / Letta — procedural vs declarative

MemGPT's three-tier model (core / archival / recall) is OS-inspired but does not distinguish procedural from declarative at the tier level. Procedural knowledge in MemGPT is **tool definitions and agent self-editing functions** — the agent calls a tool to write something to core memory, and those tools can include "this is how I should behave next time" instructions. ([Letta docs](https://docs.letta.com/concepts/memgpt), [Letta blog: Agent Memory](https://www.letta.com/blog/agent-memory))

**What to steal.**
- The pattern of **self-editing as a tool call**. Save-as-Skill should be a normal tool call from the agent's perspective, even when the user is the one initiating it.

### 1.7 Process mining and workflow induction (Aalst et al.)

A mature field on inferring process models from event logs. Algorithms include the alpha-algorithm, Heuristics Miner, Inductive Miner, and the Split Miner. They take time-stamped event sequences and emit a process model (typically a Petri net or block-structured workflow), discovering loops, parallel branches, and choices automatically. ([Aalst 2004 *Workflow Mining*](https://ieeexplore.ieee.org/document/1316839/), [Automated Discovery review, arXiv 1705.02288](https://arxiv.org/pdf/1705.02288))

**What to steal.**
- **The directly-follows graph (DFG) as a cheap precursor.** Build a DFG over the user's tool-call sequences across recent sessions; cycles and frequent paths in that graph are candidate procedures. Inductive Miner on the DFG would give us a block-structured workflow for free.
- **Frequency thresholds** for noise reduction. Process mining literature is full of guidance on filtering — typical: keep edges that appear ≥3 times across ≥2 distinct sessions.

**What to avoid.**
- The full formalism (Petri nets, conformance checking). Overkill for our scale. Use the DFG and frequency thresholds as **candidate generators**; let the LLM judge synthesize the actual procedure text.

### 1.8 Code-as-Policies / ProgPrompt / LRLL

**What it is.** Skill-as-program approaches. The LLM writes code (Python, robot-API calls) that becomes a callable function in a library; new tasks compose existing skills. LRLL ("Lifelong Robot Library Learning") explicitly builds a growing skill library through interaction. ([Code as Policies](https://code-as-policies.github.io/), [LRLL arXiv 2406.18746](https://arxiv.org/html/2406.18746))

**Relevance to us.** Our analogue isn't writing Python — it's writing AgentGraphs (our visual workflow primitive). A "skill-as-program" in our world is a saved AgentGraph in the user's library. This is candidate (C) in §3.

### 1.9 ACT-R / Anderson cognitive architecture

**Why it matters.** ACT-R formalizes the distinction we are inheriting in the memory model. Declarative memory holds **chunks** (facts); procedural memory holds **productions** (IF-condition THEN-action rules). Repeated practice **compiles** declarative knowledge into specialized productions — "automatization." This is what we want to model in the dream pass: repeated user-driven sequences get **compiled** into procedural memory by Phase 2. ([ACT-R about page](https://act-r.psy.cmu.edu/about/), [Wikipedia ACT-R](https://en.wikipedia.org/wiki/ACT-R))

**Implication.** Phase 2 of the dream pass is doing **production compilation** in ACT-R terms. The trigger ("when condition") and the action body need to be modeled separately, even if they're rendered as one paragraph for the user — this maps onto `RuleMemory.trigger` + `ProcedureMemory.steps`.

### 1.10 AutoGPT history — "abilities" and reflection

The original AutoGPT (March 2023) had a concept of "abilities" — bound capabilities like web search, file I/O, code execution — and an early issue ([#11 *Implement reflection*](https://github.com/Significant-Gravitas/AutoGPT/issues/11)) proposed reflection as a generative-improvement technique. The issue was closed not-planned. ([AutoGPT Wikipedia](https://en.wikipedia.org/wiki/AutoGPT))

**Relevance.** "Abilities" are the historical ancestor of our `Tool` and `Block`. Skills sit one layer up — a skill is a **curated combination** of abilities + instructions for when/how to use them. The lineage matters because power users in our base remember "abilities" and will expect "skills" to feel related.

### 1.11 Anthropic Managed Agents *Dreams*

Already documented in `dreaming-anthropic.md`. The thing we're imitating; the thing whose API contract our `_execute_dream_pass` shadows. Dreams write to a memory store; Anthropic does not have a separate Skill concept inside Dreams. Skills live on `claude.ai`, in Code, and on the API — **outside** Dreams. Our integration is novel: dreams *write* skills, not just memories.

---

## 2. Codebase reality check

Required verification before designing on top of `ProcedureMemory`.

### 2.1 `ProcedureMemory` is unused for writes

Grep confirms it. `autogpt_platform/backend/backend/copilot/graphiti/memory_model.py:60-81` defines `ProcedureStep` and `ProcedureMemory`. `autogpt_platform/backend/backend/copilot/tools/graphiti_store.py:104-205` accepts a `procedure` dict in the `memory_store` tool schema and persists it as part of the `MemoryEnvelope` when `memory_kind == "procedure"`. No other code writes a procedure envelope; no code reads `procedure` back out of the envelope; no `memory_kind=procedure` literal appears anywhere in the codebase.

Conclusion: **the schema is built for procedures, nothing else uses it, the field is dead weight today.** Procedure synthesis is the obvious tenant. We are not bolting onto a contested resource.

### 2.2 The MemoryEnvelope fields procedures will care about

From `memory_model.py:83-118`:

- `content: str` — one-line summary of the procedure ("Weekly newsletter publishing workflow"). Drives entity extraction and search.
- `source_kind: SourceKind` — `user_asserted` when the user runs Save-as-Skill; `assistant_derived` when the dream pass writes it.
- `scope: str` — `real:global` or `project:<name>`. A skill scoped to one project shouldn't pollute another. Per `dreaming-spec.md` §6, scope cross-leakage is a known failure mode; we inherit the mitigations.
- `memory_kind: MemoryKind.procedure` — the discriminator.
- `status: MemoryStatus` — `tentative` for fresh procedures, `active` once ratified, `superseded` on later replacement, `contradicted` if the user vetoes.
- `confidence: float | None` — LLM-rated confidence at synthesis time, 0–1.
- `provenance: str | None` — `dream:{job_id}:phase2` for dream-pass-written; `user:save_as_skill:{ts}` for user-saved.
- `procedure: ProcedureMemory | None` — the structured body. `description` + ordered `steps[]` with `action`, `tool`, `condition`, `negation`.

The shape is sufficient for **everything a procedure needs**, with one gap: there is no field for trigger conditions ("when to apply this procedure"). `RuleMemory.trigger` exists for rules; we should add an analogous concept for procedures, either as a new optional field on `ProcedureMemory` or by piggybacking on `description` with a prompt convention. Recommendation: **add `trigger: str | None` to `ProcedureMemory`**. One-line change, no migration cost (envelope is JSON-blob in episodic content).

### 2.3 Tool registration is dead simple

From `autogpt_platform/backend/backend/copilot/tools/__init__.py:62`. `TOOL_REGISTRY` is a flat `dict[str, BaseTool]`. Every tool inherits from `BaseTool` (`base.py`) and implements `name`, `description`, `parameters` (JSON schema), `requires_auth`, `_execute(user_id, session, **kwargs)`. The `as_openai_tool()` method emits the OpenAI function-calling schema. `get_available_tools()` returns the schemas for tools whose `is_available` is true.

**Adding a `save_as_skill` tool is one new class and one line in the registry.** Adding `run_skill` (to invoke a saved skill) and `find_skill` is the same. Permission gating already exists via `permissions: CopilotPermissions` on `CoPilotExecutionEntry`.

### 2.4 Blocks, AgentGraphs, and the library

From `schema.prisma`:

- `AgentGraph` (id, version, name, description, userId, optional `recommendedScheduleCron`) — a saved workflow with nodes and edges.
- `LibraryAgent` — a user's personal pinning of an AgentGraph to their library (with favorite/archive/folder).
- `AgentPreset` — an AgentGraph + a specific input configuration, optionally scheduled.
- `AgentBlock` — the registry of executable nodes.
- `StoreListing` + `StoreListingVersion` — the marketplace surface, gated by review status.

A saved skill could re-use any of these. The most natural mapping is: **a "callable" skill is an AgentGraph**, and the skill row references it. The skill's body (instructions, when-to-use) is the human/LLM-readable wrapper; the graph is the executable substrate. A "non-callable" / "advisory" skill is just text and doesn't reference a graph.

### 2.5 The chat agent's tool surface

From `dreaming-chat.md` §3. The chat agent today calls 30-odd tools including `run_agent` (executes a user's saved AgentGraph), `find_agent`, `find_library_agent`, `create_agent`. Adding skill tools fits the same surface — they're just three more entries in `TOOL_REGISTRY`.

### 2.6 Warm-context retrieval — how skills will enter context

From `dreaming-chat.md` §4 / `graphiti/context.py`. On first turn, `fetch_warm_context()` searches Graphiti for facts + last 5 episodes, formats them as `<temporal_context><FACTS>...<RECENT_EPISODES>...</temporal_context>`, and prepends them to the user message. Procedure memories already flow through this path (they're episodes with `memory_kind=procedure`). The only change for skills is: surface them **in their own section** so the LLM treats them as actionable instructions rather than as facts.

Recommendation: extend the warm-context template to include a `<SKILLS>` block holding the metadata (name + description + status + scope) of all `active` skills relevant to the current scope, similar to Anthropic's "Level 1" pre-loaded metadata. Skill bodies remain unloaded until the LLM calls a hypothetical `load_skill(name)` tool.

### 2.7 Scope semantics

From `dreaming-memory.md` §3. Scope is a property on the envelope, not a graph label. One FalkorDB database per user, all scopes live together, scope filtering happens in Python at read time. Skills inherit this; a `project:billing-app` skill is invisible from a `real:global` retrieval (unless the user explicitly invokes it). This is the right behavior — it prevents spillover between projects.

---

## 3. Three definitions of "skill" — walk-through, comparison, recommendation

The product owner's open question is whether Skill is a first-class concept. To answer it, three candidate definitions, each walked through with the same concrete example: **"Publish my weekly newsletter."**

### Candidate A — Procedure-as-memory (no execution)

**What it is.** A `MemoryEnvelope` with `memory_kind=procedure`, written into Graphiti like any other memory. Warm-context retrieval surfaces it on relevant chat turns. The chat agent **reads** the procedure and **decides** whether to follow it.

**Example data.**

```jsonc
{
  "memory_kind": "procedure",
  "scope": "project:newsletter",
  "status": "active",
  "confidence": 0.82,
  "provenance": "user:save_as_skill:2026-05-12T14:03Z",
  "content": "Weekly newsletter publishing workflow",
  "procedure": {
    "description": "Pull this week's top 5 GitHub repos, draft a Substack post, post Twitter teaser.",
    "trigger": "weekly newsletter, Friday afternoon",
    "steps": [
      { "order": 1, "action": "Search Hacker News and GitHub trending for the past 7 days", "tool": "web_fetch" },
      { "order": 2, "action": "Draft a Substack post in the user's voice using the 5 items", "tool": null },
      { "order": 3, "action": "Schedule Substack publish for Friday 9am PT", "tool": "run_agent",
        "condition": "user has approved the draft" },
      { "order": 4, "action": "Post a teaser tweet linking the post", "tool": "run_agent",
        "negation": "do not tag third-party authors without permission" }
    ]
  }
}
```

**Invocation.** None automatic. The chat agent sees the procedure in warm context, reasons about it, and asks the user before running any of the steps as live tool calls.

**User experience.** The user said "save that as a skill" and the next time they say "do my newsletter," the agent says *"I'll follow your usual workflow — pull the top 5 repos, draft a post, schedule for Friday 9am. Sound right?"* and proceeds.

**Pros.**
- Zero new infra. Already supported by `MemoryEnvelope`.
- Honest about ambiguity — the model retains judgment.
- Inherits ratification, dedup, scope, status from the existing memory lifecycle.

**Cons.**
- No "I want to invoke this directly without natural-language preamble" UX. The user can't say "/run newsletter-workflow"; they have to ask the agent to follow the procedure.
- Not discoverable as a library. There's no list-skills page; skills live inside graph nodes.
- No sharing primitive. The marketplace doesn't know about it.

### Candidate B — Skill-as-prompt-fragment (`SKILL.md`-style)

**What it is.** A curated `SKILL.md`-shaped document — frontmatter with `name`/`description` + markdown body — stored in a new lightweight table or in `WorkspaceManager`. Surfaced to the chat agent the same way Anthropic does: metadata in the system prompt; body fetched on-demand by a `load_skill(name)` tool call. Trigger conditions in the description; behavior in the body.

**Example data.**

```yaml
---
name: weekly-newsletter
description: Publish the user's weekly newsletter. Use when the user asks to do their newsletter, on Friday afternoons, or mentions "this week's repos".
scope: project:newsletter
status: active
version: 3
---

# Weekly newsletter publishing

## Steps

1. Fetch top 5 trending GitHub repos and Hacker News stories from the past 7 days.
2. Draft a Substack post in the user's voice (see VOICE.md if uncertain).
3. Confirm the draft with the user before publishing.
4. Schedule Substack publish for Friday 9am PT via run_agent.
5. Post a teaser tweet linking the post via run_agent.

**Never** tag third-party authors without explicit user permission.

## See also

- VOICE.md for the user's writing-voice rules.
```

**Invocation.** The chat agent matches the description against the current request, calls `load_skill("weekly-newsletter")`, gets the body into context, then executes the steps using normal tool calls.

**User experience.** The skill is browsable in the library UI. The user can hit "edit" and modify the body directly. The agent treats the body as authoritative instructions for the task.

**Pros.**
- Industry-standard shape. Power users already know it. Compatible with Anthropic, LangChain, agentskills.io.
- Cleanly discoverable. There's an obvious UI ("Skills" tab in the library).
- Body is human-readable and human-editable.
- Progressive disclosure is easy and well-studied.

**Cons.**
- We invent a small parallel store for `SKILL.md` content vs. the existing Graphiti store. (Mitigated below.)
- "Where does the executable part live?" — the body says "run_agent with these args" but doesn't bind to a specific saved AgentGraph. Up to the LLM to map.

### Candidate C — Skill-as-callable-agent (AgentGraph)

**What it is.** A skill is a saved `AgentGraph` in the user's library. It already has a name, description, version, inputs, outputs, optional cron. The chat agent invokes it via the existing `run_agent` tool. The Save-as-Skill flow builds the graph from the procedure steps.

**Example data.** An `AgentGraph` named "Weekly Newsletter Publisher" with nodes (`web_search` → `llm_draft` → `human_approval` → `substack_schedule` → `twitter_post`) and edges, owned by the user, surfaced in the library.

**Invocation.** `run_agent(library_agent_id=..., inputs={"topic_hint": "AI tooling"})`. Fully deterministic. Same surface as every other agent in the platform.

**User experience.** The skill is a normal agent. The user can edit it in the visual builder, schedule it, fork it, share it via the marketplace.

**Pros.**
- Reuses agent runtime entirely. Zero new execution infra. Cron, scheduling, library, marketplace already exist.
- Most natural fit for **multi-step, parameterized, repeatable** workflows.
- The marketplace path is "already done" for free.

**Cons.**
- Heavy. A trivial "always reply in French to project:foo" preference doesn't deserve a graph.
- Graphs are deterministic and don't fluidly degrade. If a step fails the whole graph fails; advisory skills handle this more gracefully.
- The "when to use" judgment is on the LLM choosing whether to call `run_agent`, not embedded in the graph itself.

### Comparison

| Axis | (A) Procedure memory | (B) SKILL.md fragment | (C) AgentGraph |
|---|---|---|---|
| New infra cost | None | Small new table + 1 tool | None |
| Discoverability (UI) | Bad | Good | Excellent |
| Sharing/marketplace | None | New endpoint required | Already exists |
| Executable | LLM decides per-step | LLM decides per-step | Deterministic |
| Editing | Edit a memory envelope | Edit markdown body | Visual graph builder |
| Right for trivial preferences ("always X") | Maybe | Yes | No |
| Right for complex multi-step workflows | Maybe | Yes | Yes |
| Versioning | Memory `status` lifecycle | Need version field | `AgentGraph.version` exists |
| Sandboxing for shared skills | N/A | Inspect markdown | Block-level allowlist already exists |
| Compatibility with Anthropic/LangChain | None | Native | None |

### Recommendation: **all three, with a single binding row**

The mistake would be picking one. They serve different procedure shapes. The right design is:

1. Introduce a **`Skill` row in Postgres** that is the single user-facing identity. The row holds `name`, `description`, `body` (the markdown), `triggers` (denormalized for fast scan), `parameters` schema, `scope`, `status`, `version`, `visibility`, `owner_id`.
2. The row optionally **points to a substrate**:
   - `procedure_envelope_uuid: str | None` — if present, the canonical procedure body lives here (Candidate A retained for legacy / migrating).
   - `agent_graph_id, agent_graph_version: str?, int?` — if present, the skill is callable as an agent (Candidate C).
   - Else, the skill is body-only (Candidate B).
3. Retrieval, sharing, versioning, and UI all key off the `Skill` row. The substrate is an implementation detail of *how the skill executes*, not *what the skill is*.

This means: **Skill is a first-class object, but it's a thin row.** It composes existing primitives rather than replacing them. We get the UX and discoverability win without inventing a new runtime.

---

## 4. Pattern detection from chat history

How does the dream pass actually find a procedure worth saving?

### 4.1 The candidate generators

Three signals, run in parallel during Phase 2 of the dream pass:

**(a) Tool-call sequence repetition (process-mining style).**
Build a directly-follows graph (DFG) over each session's tool calls. Each node is a tool name; each edge is a transition with a count. Aggregate across the user's last N sessions. Edges that appear ≥3 times across ≥2 sessions become candidate procedure backbones. Cycles in the DFG that recur in this way are strong candidates. The Inductive Miner algorithm (Aalst et al.) can extract a block-structured workflow from the DFG in linear time; we ship that off-the-shelf or implement the simpler alpha-algorithm. ([Aalst 2004](https://ieeexplore.ieee.org/document/1316839/), [Discovery review](https://arxiv.org/pdf/1705.02288))

**(b) Near-duplicate user-message clustering.**
Embed user messages (we already embed with `text-embedding-3-small` via the Graphiti embedder; see `dreaming-memory.md` §7). Cluster with HDBSCAN or simple k-means at a fixed similarity threshold (cosine ≥ 0.85). Clusters with ≥3 members across ≥2 sessions are candidate trigger phrases. The cluster centroid's nearest-neighbor message becomes the canonical trigger description.

**(c) Recurring agent-graph invocations.**
The simplest signal. If the user runs the same `library_agent_id` with similar inputs ≥3 times, that's a candidate skill — at minimum a "use this agent more readily" rule, often a parameterized procedure.

### 4.2 The LLM judge

Candidate generators are cheap and noisy. Phase 2 of the dream pass feeds the top-K (~20) candidates to Claude Opus 4.7 with extended thinking (`dreaming-spec.md` §1), structured-output, asking it to:

1. Reject candidates that are one-off coincidences or already covered by an existing skill (dedup against `Skill` table + `ProcedureMemory` envelopes).
2. For survivors, draft a `Skill`-shaped JSON: name (gerund-form, `processing-pdfs`-style — Anthropic best practice), description (when-to-use + what-it-does), trigger conditions, scope, body steps, confidence.
3. Suggest parameters where the candidate's input varied across observations.

Output: `proposed_skill` envelopes with `status=tentative`, `provenance="dream:{job_id}:phase2"`. They flow through the existing ratification loop:

- Surfaced to the user in the dream-diff UI (P6/P7 on the roadmap).
- The user accepts → `status=active`, the `Skill` row gets created.
- The user vetoes → `status=contradicted`, fed back into the "don't propose this again" filter.
- 30 days unratified → `status=superseded` with reason `unratified`.

### 4.3 Eval methodology for the synthesis pass

- **Precision.** Sample 50 dream-proposed skills/week. Manual review (or a separate judge LLM) labels each as `good`, `redundant`, `wrong`, `noise`. Target ≥60% `good` in the first month, ≥75% by month 3.
- **Coverage.** Of the user's actually-recurring workflows (defined as ≥3-occurrence tool sequences in the DFG), what fraction does the dream propose? Sampled monthly via stratified audit.
- **Time to first useful skill.** From a user's first chat session, how many days until a `Skill` reaches `status=active`? Target p50 ≤ 14 days, p95 ≤ 30 days.
- **Inter-skill overlap.** Are we generating near-duplicates? Compute pairwise cosine similarity of skill descriptions; alert on >0.85 pairs that are not flagged as duplicates by the dedup pass (P2).

### 4.4 Why hybrid

Heuristics-only misses everything except literal tool-call repetition; LLM-only is wasteful (Phase 2 cost balloons if it scans hundreds of sessions raw) and biased (LLMs over-detect "patterns" in noise). The DFG plus message-cluster plus agent-replay candidates give the LLM judge exactly the structure it needs to confirm and elaborate. This mirrors how Voyager works — heuristic curriculum proposes tasks; LLM critic validates skills. ([Voyager](https://voyager.minedojo.org/))

---

## 5. The Save-as-Skill tool — full spec

### 5.1 Tool signature

```python
class SaveAsSkillTool(BaseTool):
    name = "save_as_skill"
    description = (
        "Save the current workflow as a reusable skill. The skill is stored "
        "in the user's skill library and can be invoked later by name. "
        "Use when the user asks to 'save this', 'make a skill', 'add to my "
        "library', or when a procedure has been demonstrated and confirmed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "kebab-case identifier, ≤64 chars, lowercase + hyphens, gerund preferred." },
            "description": { "type": "string", "description": "What it does AND when to use it. 1-3 sentences." },
            "body": { "type": "string", "description": "Markdown procedure body. May reference workspace files." },
            "scope": { "type": "string", "description": "real:global, project:<name>, etc. Default: inherit from current session." },
            "parameters": { "type": "object", "description": "Optional JSON-schema of named parameters the skill accepts." },
            "agent_graph_id": { "type": "string", "description": "Optional. Bind the skill to an existing agent graph in the user's library." },
            "agent_graph_version": { "type": "integer" },
            "visibility": { "type": "string", "enum": ["private", "shareable", "marketplace_pending"], "default": "private" },
            "source_evidence": { "type": "array", "items": { "type": "string" }, "description": "Session/message refs the skill was synthesized from." }
        },
        "required": ["name", "description", "body"]
    }
    requires_auth = True
```

Return: `SkillSavedResponse` with the new skill's id, version, and a frontend deep-link to its detail page.

### 5.2 Trigger surfaces

Five ways a skill gets created. All converge on the tool.

1. **User says "save that as a skill" in chat.** The chat agent calls `save_as_skill` with extracted name/description/body. The agent should pre-draft and ask the user to confirm before persisting.
2. **User clicks "Save as Skill" button on a chat message in the UI.** Sends a synthetic user turn `"save the procedure from message #N as a skill"`. The tool runs; the result surfaces in the same conversation.
3. **User runs `/save-skill` slash command.** Same as button.
4. **Dream pass accepts a proposed procedure.** The synthesis path writes `tentative` proposals; the user ratifies via the dream-diff UI (P6/P7); ratification promotes via `save_as_skill` under the hood with `source_kind=assistant_derived`, `provenance="dream:{job_id}"`.
5. **Library UI explicit author flow.** "New Skill" button → form-based authoring → calls `save_as_skill` directly.

### 5.3 What the user can edit

- **At creation:** name, description, scope, body, parameters, visibility, optional graph binding.
- **Post-creation:** all of the above, plus `status` (active ↔ paused), version. Each save creates a new `Skill` row with incremented `version` and `parentSkillId` link; old version retained (audit + rollback). This mirrors `AgentGraph` versioning (which uses a composite primary key `[id, version]`).
- **Triggers:** the description field doubles as the trigger surface. Anthropic's design works because the description is rich enough to drive selection. We follow the same convention.

### 5.4 Server-side behavior

1. **Validate** name (regex `^[a-z0-9-]{1,64}$`, no reserved words: `dream`, `system`, `autogpt`, plus Anthropic's `anthropic`/`claude` since our exports should round-trip cleanly).
2. **Dedupe.** Embed the description, search existing skills (vector + name exact-match); if a near-duplicate exists, return `SkillExistsResponse` and let the user choose: replace, version-bump, or cancel.
3. **Write the `Skill` row** (Postgres). Visibility `private` by default.
4. **Write a `MemoryEnvelope` mirror** to Graphiti with `memory_kind=procedure`, populated `ProcedureMemory`, and `provenance` linking back to the `Skill.id`. This keeps warm-context retrieval working without a second retrieval path.
5. **Emit an SSE event** (`StreamSkillSaved`) so the UI updates the library inline.
6. **Track** a `skill_saved` analytics event with source (user/dream/library), scope, has_graph_binding.

### 5.5 Visibility model

Three states; the model deliberately rules out the "anyone with a link" semi-public state Anthropic's `claude.ai` uses, because it conflicts with our marketplace review process.

- **private** — only the owner; default. Used by warm-context retrieval, callable from the owner's chat.
- **shareable** — the owner can grant access to specific other users (org-scoped when orgs exist; see "Deferred" in `dream/TODO.md`). Not listed in the marketplace, not searchable across the platform.
- **marketplace_pending** → **marketplace_listed** — submitted for review; on approval, the skill gets a `StoreListing` row (existing table) and is browsable in the marketplace. Reuses the existing store review workflow (`SubmissionStatus` enum in `schema.prisma:1197`).

### 5.6 Trust model for shared/marketplace skills

A shared skill that points at an `AgentGraph` inherits the graph's existing trust gates: blocks with `requires_auth` prompt the consumer for credentials at first run, no graph can run unattended on a third party's account. A shared skill that is **body-only** (no graph) is **advisory text**: the LLM reads it, decides whether to apply it, can disregard it. We do NOT yet support **executable scripts inside a skill body** (Anthropic does; ours doesn't). When/if we add bundled scripts, those run in the workspace sandbox (`WorkspaceManager` already enforces session-scoped isolation), not on the consumer's host.

### 5.7 Lifecycle

```
draft (frontend-only) → tentative (saved, not yet validated)
                       ↘
                        active (ratified by use or explicit promote)
                       ↗
                       superseded (replaced by a newer version)
                       contradicted (user-rejected, dream proposal filtered out)
                       archived (user-shelved)
```

Transitions follow the existing `MemoryStatus` semantics (`dreaming-spec.md` §5). A skill auto-promotes from `tentative` to `active` on first successful invocation by the chat agent (warm-context retrieval surfaced it and the agent followed the body) OR explicit user click.

### 5.8 End-to-end lifecycle example

Time T0: user runs the same three-step "summarize my Linear sprint, draft a Slack post, schedule for Monday 9am" workflow on three Fridays in a row, manually each time.

Time T0 + 7d: Friday-night dream pass. Phase 2 DFG over the user's last 14 days finds the 3-cycle. LLM judge confirms; writes a `proposed_skill` envelope with name `summarizing-sprint-recap`, status `tentative`, confidence 0.71.

Time T0 + 8d (Saturday morning): user opens the dream-diff UI (P6/P7), sees "Proposed skill: summarizing-sprint-recap — confirmed in 3 of last 4 weeks. Accept / Edit / Reject?" User clicks **Edit**, renames to `weekly-sprint-recap`, tightens the description.

User clicks **Accept**. The `Skill` row is written, `status=active`, `version=1`, visibility=`private`. The Graphiti mirror envelope is also active. The chat agent's `<SKILLS>` warm-context block now includes the skill metadata for `project:work`.

Time T0 + 12d (next Friday): user says "do the sprint recap". The chat agent sees the skill metadata in warm context, calls `load_skill("weekly-sprint-recap")`, gets the body, walks the user through the three steps using normal tool calls. The skill is invoked → analytics records a `skill_invoked` event → ratification confirms `active` status if it wasn't already.

Time T0 + 60d: user clicks "Share with team" in the skill detail UI. Visibility flips to `shareable`. (Marketplace listing is a separate, reviewed flow.)

---

## 6. Skills as first-class objects? Defended answer

The three options the brief asks for:

### Option 1 — Yes: full first-class `Skill` model

Pros: cleanest UI, sharing, versioning, marketplace path. Cons: more schema and code; risk of duplicating things `AgentGraph` already provides.

### Option 2 — No: `ProcedureMemory` carries everything

Pros: zero schema cost. Cons: no library UI ("how do I see all my skills?"), no native sharing, retrieval works but discoverability is poor, no version control beyond the memory status lifecycle.

### Option 3 — Hybrid: in-graph for private, promote to Postgres `Skill` only on publish

Pros: defers schema until usage justifies it. Cons: two retrieval paths, two sources of truth, harder to surface in UI consistently.

### Recommendation: Option 1, *thinned* — a single Postgres `Skill` row that points at existing substrates rather than replacing them.

The win is **identity and addressability**. A `Skill.id` is what the UI, the marketplace, the dream-diff artifact, and analytics agree on. Underneath, the skill's behavior can be a Graphiti envelope, an AgentGraph, or pure markdown. The row is small (~10 columns) and inexpensive to ship.

Why not Option 2: I tried walking through the UX. "How do I see all my skills?" without a table requires either a denormalized query against episodic content JSON (slow and brittle) or a derived view (which is essentially Option 1 by another name). Skipping the row saves no work; it just moves the work into ad-hoc query code.

Why not Option 3: two sources of truth always produce drift. The schema-deferral argument is real but the cost of an early `Skill` table is small (~30 lines of Prisma + migration). Speculation: I expect we'd be paying down Option 3 within a quarter.

**The proposed Prisma model:**

```prisma
model Skill {
  id                  String   @id @default(uuid())
  createdAt           DateTime @default(now())
  updatedAt           DateTime @default(now()) @updatedAt

  userId              String
  User                User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  name                String   // kebab-case, lowercase
  description         String
  body                String   // markdown
  scope               String   @default("real:global")

  // Optional bindings to existing substrates
  procedureEnvelopeUuid String?
  agentGraphId          String?
  agentGraphVersion     Int?
  AgentGraph            AgentGraph? @relation(fields: [agentGraphId, agentGraphVersion], references: [id, version], onDelete: SetNull)

  parameters          Json     @default("{}")   // optional schema
  triggers            Json     @default("[]")   // denormalized trigger phrases for fast scan

  status              String   @default("tentative")  // mirrors MemoryStatus
  confidence          Float?
  provenance          String?

  visibility          String   @default("private")    // private | shareable | marketplace_pending | marketplace_listed
  version             Int      @default(1)
  parentSkillId       String?
  Parent              Skill?   @relation("SkillVersions", fields: [parentSkillId], references: [id])
  Children            Skill[]  @relation("SkillVersions")

  isArchived          Boolean  @default(false)
  isDeleted           Boolean  @default(false)

  @@unique([userId, name, version])
  @@index([userId, status, isArchived])
  @@index([userId, scope])
  @@index([agentGraphId, agentGraphVersion])
}
```

Cost: one migration, one Prisma model, ~6 endpoints (list, get, create, update, archive, publish-to-marketplace). Maps onto the existing `LibraryAgent` / `StoreListing` patterns exactly.

---

## 7. User-uploadable skills

The product owner asked for this specifically.

### 7.1 Format

The right format is **a `SKILL.md` file plus an optional bundle directory**. Concretely: a `.zip` upload containing at minimum `SKILL.md`, optionally `STEPS.md`, `EXAMPLES.md`, `parameters.json`, and reference docs the body links to. This is exactly Anthropic's format. ([Anthropic Skills overview](https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/overview))

A single-file alternative (`.md` with YAML frontmatter) for the common case. The parser accepts either.

Upload endpoint: `POST /v2/skills/import` accepting `multipart/form-data` with `file` (the zip or .md) and `visibility` (default `private`). The server validates frontmatter, runs the same dedup/validation as Save-as-Skill, and writes the `Skill` row.

### 7.2 Trust model for uploads

An uploaded skill is **identical** to one synthesized from chat — same `Skill` row, same retrieval path, same lifecycle. **No quarantine**, because the skill body is advisory text that the LLM reasons about before acting; it cannot exfiltrate data on its own. The only elevated trust gate is at *invocation*: if the skill body says "run agent X", the chat agent still calls `run_agent` with all the normal credential/permission checks.

**Caveat:** if/when we support executable scripts inside a skill bundle (we don't in v1), upload skills must run those in the workspace sandbox only, never on the consumer's host or with their credentials. See §5.6.

### 7.3 Sharing

Upload defaults to `private`. The user can flip visibility to `shareable` or submit to `marketplace_pending` from the skill detail UI — the same path as a chat-synthesized or dream-synthesized skill.

### 7.4 Importing from the Anthropic ecosystem

Feasibility: high. The `SKILL.md` shape is identical; we just need to drop frontmatter fields we don't recognize (e.g. `license`, `compatibility`) into a JSON `metadata` blob and preserve them on re-export. Importing scripts is **out of scope for v1** — we strip them on import with a warning, and the imported skill becomes body-only.

Usefulness: meaningful in two directions.

- Inbound (Anthropic → us): power users may have a personal skill library in `~/.claude/skills/`. Letting them upload that library in one shot is a big onboarding win for the segment most likely to use the dream feature.
- Outbound (us → Anthropic): when the user wants to use their AutoGPT-synthesized skills in Claude Code, exporting as `SKILL.md` makes that one click.

We should ship both directions in v1.

---

## 8. Integration with the dream loop

### 8.1 Where procedure synthesis lives in the pipeline

Phase 2 (`dreaming-spec.md` §1 — recombination, Opus 4.7, extended thinking). Procedures are a category of `proposed_finding` with a richer schema. Phase 3 (sanitize & commit) inherits all its existing checks: drop on duplicate-of-existing, hallucinated entities, scope leakage, persona drift. Add one new Phase 3 check: **drop on `name` collision with an existing active skill** (unless the new one's confidence ≥0.1 above the existing).

### 8.2 How skills enter chat context

Two mechanisms, complementary:

- **Skill metadata in warm context.** Extend `fetch_warm_context()` (`graphiti/context.py:20`) to add a `<SKILLS>` block listing the metadata (name, description, scope, status) of active skills relevant to the session's scope. The block is bounded — top-K by description-embedding similarity to the first user message — to keep the budget tight.
- **`load_skill(name)` tool.** When the chat agent decides to apply a skill, it calls this tool to get the full body into context. Cost: one tool round-trip, ~500–2000 tokens of skill body.

This is exactly Anthropic's progressive disclosure pattern, mapped onto our tool/warm-context surface.

### 8.3 Ratification

Already covered in `dreaming-spec.md` §5 for memories generically. For skills specifically:
- An auto-extracted skill starts `tentative`.
- First successful invocation by the chat agent (warm-context-surfaced + body-loaded + at least one step run without contradiction) → `active`.
- 30 days unused with no acceptance → `superseded`/`unratified`.
- User-uploaded or user-Save-as-Skill skills start `active` directly — the user authoring it *is* the ratification signal.

### 8.4 Failure modes specific to skills

| Failure | Mitigation |
|---|---|
| **One-off mistake gets durably memorized.** User did something wrong once, dream proposes a skill capturing the wrong thing. | Confidence threshold: dream proposals at `confidence < 0.6` go to `tentative` and require user ratification before any retrieval. Dedup against existing active skills before proposing. |
| **Skill creep.** User accumulates 200 skills, warm-context budget is overrun. | Cap warm-context skill block at top-K by relevance (K=5). Archive button + auto-archive after 90 days of zero invocation. |
| **Stale skill.** "Always use gpt-image-2 for thumbnails" becomes wrong after gpt-image-2 is retired. | Skills are subject to the same fact-rot demotion (P0.3a, P0.5 web-verified). The `web_fact_check` tool runs over skill bodies during the dream pass. |
| **Cross-scope leakage.** A `project:foo` skill surfaces in `project:bar`. | Same `_filter_episodes_by_scope` mechanism. Skills are scope-tagged. |
| **Hostile shared skill.** A marketplace-listed skill instructs the consumer's chat agent to exfiltrate data. | Body is advisory; tool calls still require user-side credentials and permissions. Future: a "review skill body" step before adopting a shared skill, analogous to the AgentGraph review for store listings. |
| **Skill-on-skill recursion.** Skill A references skill B references skill A. | Bound recursion depth at 3; cycle-detect at load time. |
| **Pattern detection over-fits noise.** The user did the same thing twice by coincidence. | Threshold: ≥3 occurrences across ≥2 distinct sessions to make it past the heuristic candidate generator. |

---

## 9. Evaluation and metrics

In addition to the synthesis-specific metrics in §4.3:

- **Save-as-Skill conversion.** When the chat agent suggests "Save this as a skill", what fraction of users accept? Sample weekly; alert if this drops below 30% (suggests over-eager prompting) or above 80% (suggests under-prompting).
- **Skill invocation rate.** Of `active` skills, what fraction get invoked within 30 days? Target ≥50%. Skills with zero invocations after 30 days are candidates for archive.
- **Skill body length distribution.** Median, p90 of body markdown length. If p90 exceeds ~3000 tokens, the progressive-disclosure assumption is breaking; likely a signal we need separate reference files (Anthropic best practice).
- **Cross-skill description overlap.** Pairwise cosine similarity of descriptions; track >0.85 pairs. Should converge with the dedup pass (P2).
- **Marketplace conversion.** Of `shareable` skills, what fraction submit for marketplace listing? Of those, what fraction approve? Of approved, what fraction sustain ≥10 invocations across other users in 30 days?

A single eval-suite addition to the P0.6 memory-benchmark harness: a curated set of "procedure-rich" anonymized user transcripts where the synthesis pass should produce ≥1 high-quality skill. Run weekly post-deploy; gate releases on no-regression.

---

## 10. Build plan

Concrete files to add/edit, ordered for incremental shippability. Estimates are engineer-days for someone fluent in the backend.

### v1 — Save-as-Skill exists, dream synthesizes procedures, warm context surfaces skills. ~3 engineer-weeks.

1. **Schema** (~1d).
   - Add `Skill` Prisma model (§6). Migration.
   - Optional: add `trigger: str | None` to `ProcedureMemory` in `memory_model.py:60-81`. JSON-blob field; no migration needed.

2. **Skill DB accessor** (~1d).
   - New file: `backend/data/db_accessors/skill.py`. CRUD with `userId` ownership checks (per AGENTS.md PR review rule on `data/*.py`).
   - Versioning helper: `create_new_version(skill_id, ...)` clones the row with incremented `version` and `parentSkillId`.

3. **API endpoints** (~2d).
   - New router: `backend/api/features/skills/routes.py`. `POST /v2/skills`, `GET /v2/skills`, `GET /v2/skills/{id}`, `PATCH /v2/skills/{id}`, `POST /v2/skills/{id}/archive`, `POST /v2/skills/{id}/publish`, `POST /v2/skills/import`.
   - Pydantic models alongside; integrate with `mock_jwt_user`-style auth.
   - Tests in `routes_test.py` with snapshot assertions per TESTING.md.

4. **`save_as_skill` tool** (~1d).
   - New file: `backend/copilot/tools/save_as_skill.py`. Inherits `BaseTool`. Wires into `TOOL_REGISTRY`.
   - Calls the Skill DB accessor; also writes a Graphiti mirror envelope via `enqueue_episode()`.
   - Tests: `save_as_skill_test.py`.

5. **`load_skill` and `find_skill` tools** (~1d).
   - `load_skill(name) → SkillBodyResponse` for full-body fetch on invocation.
   - `find_skill(query, scope) → list[SkillMetadata]` for explicit search.
   - Register in `TOOL_REGISTRY`.

6. **Warm-context integration** (~2d).
   - Extend `fetch_warm_context()` in `graphiti/context.py:20` to add a `<SKILLS>` block.
   - Top-K-by-description-embedding similarity; scope-filtered; capped at 5 or a token budget.
   - Update prompt construction in `service.py:360` (`inject_user_context`) to include the block.
   - Tests for context assembly.

7. **Phase 2 dream-pass procedure synthesis** (~3d).
   - In the eventual `backend/executor/dream_pass.py` (introduced in P0):
     - Candidate generator: DFG over recent sessions' tool calls + message-cluster + agent-replay (§4.1).
     - LLM judge prompt fragment, structured-output schema for `proposed_skill`.
   - Phase 3 sanitization additions: drop on `name` collision, drop on `confidence < 0.4`, drop on duplicate-of-existing.

8. **Frontend — Skill list and detail pages** (~3d).
   - New route: `src/app/(platform)/skills/page.tsx` + `useSkillsPage.ts`.
   - Detail: `src/app/(platform)/skills/[id]/page.tsx`.
   - "Save as Skill" chat-message-context-menu button — call the tool via the existing tool-invocation path.
   - Integration tests per `frontend/TESTING.md`.

9. **SSE event** (~0.5d).
   - `StreamSkillSaved` in `response_model.py`, registered in `stream_registry._reconstruct_chunk`. UI updates inline.

10. **Telemetry** (~0.5d).
    - `skill_saved`, `skill_invoked`, `skill_archived`, `skill_published` analytics events.

Total: ~3 engineer-weeks for one engineer; ~2 with two engineers working frontend/backend in parallel.

### v2 — Sharing, marketplace, upload, import. ~2 engineer-weeks.

11. **Upload endpoint and parser** (~1d). Zip-with-`SKILL.md` plus single-`.md` paths. Strip Anthropic-incompatible fields into a `metadata` blob.
12. **Export endpoint** (~0.5d). Symmetrical: emit a `SKILL.md` (or zip) from a `Skill` row.
13. **Sharing model** (~2d). `shareable` visibility, per-user grants, view-only embed of skill detail. Defer fine-grained ACLs until orgs ship.
14. **Marketplace integration** (~3d). New `StoreListing` row when a skill is published. Skill detail page shows install button for consumers; install copies the skill row into the consumer's account (with `parentSkillId` retained for attribution).
15. **Dream-diff UI integration** (~3d). When P6/P7 lands, the dream-diff artifact includes proposed skills with Accept/Edit/Reject affordances.

### Deferred — earned in by usage signal

- **Executable scripts inside skill bundles.** Wait for clear user demand and a sandbox we trust (E2B or the workspace `bubblewrap` path). Speculation: this is where Anthropic's "skill" shape really pays off; we'll likely want it within 6 months of v1.
- **Cross-skill composition graphs.** A "meta-skill" that calls two other skills. Defer until we observe users hand-composing them in chat.
- **Org-shared skills.** Requires org primitives first.
- **Skill recommendation feed.** "Other users with similar workflows saved this skill" — needs a privacy review and cross-user signal aggregation. Park for now.

---

## 11. Loose ends and open questions

1. **Naming.** `Skill` vs. `Procedure`. The brief uses both. Recommend `Skill` externally (industry standard, less academic), keep `ProcedureMemory` as the internal substrate name. They're aligned on `Skill.procedureEnvelopeUuid`.

2. **Versioning user-edits.** Should every edit bump version, or only "structural" edits? Recommend: bump on body change ≥10% (by line diff). Trivial typo fixes do not bump. Implementation: a soft heuristic in the save endpoint; the user can force-bump if they want a clean cut.

3. **What about preferences?** `MemoryKind.preference` exists. Is "always reply in French" a skill or a preference? Recommend: preferences are facts about the user, skills are workflows the user does. The rule of thumb: if it has steps, it's a skill; if it's a constant, it's a preference. The dream pass can write both; the same envelope shape carries them.

4. **Skill triggers vs. RuleMemory.** A skill's trigger condition overlaps with `RuleMemory.trigger`. Should we converge them? Recommend yes, eventually — a skill's `triggers` JSON could be `RuleMemory[]`. Defer the type unification to v2 to avoid blocking shipping.

5. **Cost.** Phase 2 procedure synthesis runs Opus 4.7 — already in the budget. The candidate generator is CPU-light. Per-user dream-pass cost should rise by <5% with procedure synthesis added; if it spikes, the DFG fanout is the suspect.

6. **What if a user disables dreaming?** Save-as-Skill still works (it's a synchronous chat tool, not part of the dream pass). Pattern detection doesn't run. The user can still manually author and upload.

7. **Speculation: the marketplace will out-pace expectations.** If skills land cleanly, "I have an AutoGPT skill that…" becomes a content category the way Custom GPTs did. We should be ready with discovery surfaces (categories, search) earlier than the conservative roadmap suggests.

---

## 12. Summary table — what we proposed, in one screen

| Question | Answer |
|---|---|
| What is a skill in our system? | A Postgres `Skill` row pointing at one of: a `ProcedureMemory` envelope, an `AgentGraph`, or pure markdown body. Identity lives in the row; behavior lives in the substrate. |
| Is `Skill` first-class? | Yes — but as a thin row, not a parallel runtime. It composes existing primitives. |
| How are skills authored? | (a) `save_as_skill` tool in chat, (b) library UI form, (c) slash command, (d) dream pass synthesis with user ratification, (e) zip/`.md` upload. |
| How are skills detected from chat? | DFG over tool calls + user-message embedding clusters + agent-replay candidates → LLM judge confirms and structures → `proposed_skill` envelope → ratification loop. |
| Where do skills enter chat context? | Metadata in warm-context `<SKILLS>` block on first turn; full body on-demand via `load_skill(name)` tool. Progressive disclosure, Anthropic-style. |
| What's the file format on disk / on export? | Anthropic-compatible `SKILL.md` with YAML frontmatter (`name`, `description`, optional metadata) + markdown body. Optional bundle directory for reference files. |
| Sharing model? | `private` → `shareable` (named grants) → `marketplace_pending` → `marketplace_listed` via existing store review workflow. |
| What about uploaded skills? | Same `Skill` row as synthesized ones. No quarantine — bodies are advisory text. Anthropic-format zip/`.md` accepted. |
| What about hostile skills? | Body is advisory; tool calls still require user credentials. Marketplace listing gated by existing review. Future: skill-body review step at install time. |
| Build cost? | v1 ~3 engineer-weeks (Save-as-Skill + dream synthesis + warm-context integration). v2 ~2 engineer-weeks (upload, share, marketplace). |
| First-class enough for the marketing copy? | Yes — "Skills" is the user-facing name. "Procedures" is the internal/research term. |

---

## Cited sources

- Anthropic Agent Skills overview — <https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/overview>
- Anthropic Agent Skills best practices — <https://platform.claude.com/docs/en/docs/agents-and-tools/agent-skills/best-practices>
- Anthropic Skills launch post — <https://claude.com/blog/skills>
- Agent Skills open standard — <https://agentskills.io>
- Voyager: An Open-Ended Embodied Agent with LLMs — <https://voyager.minedojo.org/>, <https://arxiv.org/abs/2305.16291>
- LangChain Skills — <https://www.langchain.com/blog/langchain-skills>
- LangChain Deep Agents Skills docs — <https://docs.langchain.com/oss/python/deepagents/skills>
- LangMem SDK launch — <https://blog.langchain.com/langmem-sdk-launch/>
- LangMem concepts — <https://langchain-ai.github.io/langmem/concepts/conceptual_guide/>
- Letta / MemGPT docs — <https://docs.letta.com/concepts/memgpt>, <https://docs.letta.com/concepts/letta/>
- Letta blog: Agent Memory — <https://www.letta.com/blog/agent-memory>
- OpenAI Custom GPTs help — <https://help.openai.com/en/articles/8554397-creating-and-editing-gpts>, <https://openai.com/index/introducing-gpts/>
- OpenAI GPT Actions — <https://developers.openai.com/api/docs/actions/introduction>
- Aalst — Workflow Mining: Discovering Process Models from Event Logs — <https://ieeexplore.ieee.org/document/1316839/>, <https://www.vdaalst.com/publications/p245.pdf>
- Automated Discovery of Process Models from Event Logs (review) — <https://arxiv.org/pdf/1705.02288>
- Code as Policies — <https://code-as-policies.github.io/>, <https://arxiv.org/pdf/2209.07753>
- LRLL — Lifelong Robot Library Learning — <https://arxiv.org/html/2406.18746>
- ACT-R — <https://act-r.psy.cmu.edu/about/>, <https://en.wikipedia.org/wiki/ACT-R>
- AutoGPT history — <https://en.wikipedia.org/wiki/AutoGPT>, <https://github.com/Significant-Gravitas/AutoGPT/issues/11>, <https://github.com/Significant-Gravitas/AutoGPT/issues/15>
- Anthropic Managed Agents Dreams — <https://platform.claude.com/docs/en/managed-agents/dreams>, <https://claude.com/blog/new-in-claude-managed-agents>
