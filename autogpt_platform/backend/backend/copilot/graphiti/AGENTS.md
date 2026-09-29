# Graphiti Memory

This directory contains the Graphiti-backed memory integration for CoPilot.
This file is developer documentation only — it is NOT injected into LLM prompts.
Runtime prompt instructions live in `prompting.py:get_graphiti_supplement()`.

## Scope

- Keep Graphiti and FalkorDB-specific logic in this package.
- Prefer changes here over scattering Graphiti behavior across unrelated copilot modules.

## Debugging

- Use raw FalkorDB queries to inspect stored nodes, episodes, and `RELATES_TO` facts before changing retrieval behavior.
- Distinguish user-provided facts, assistant-generated findings, and provenance/meta entities when evaluating memory quality.

## Design Intent

- Preserve per-user isolation through `group_id`-scoped databases and clients.
- Build a `MemoryScope` (`scope.py`) once where a request enters the memory code and pass it down; read `group_id` / `scope_key` / `redis_key(...)` off it rather than re-deriving them from a `(user_id, expert_id)` pair, and open drivers with `open_driver(scope)`.
- Be careful about memory pollution from assistant/tool phrasing; extraction quality matters as much as ingestion success.
- Keep warm-context and tool-driven recall resilient: failures should degrade gracefully rather than break chat execution.
- Recall goes through `recall.py` (what may be read back) and `recall_render.py` (how it is written out). Read facts with `search_facts` and episodes with `recent_episodes`; what the assistant is shown is read again by uuid as the last graph read before rendering (`search_facts` ends with `live_now`, and warm context and `memory_search`, which hold both lists while the slower read finishes, call `recall_recheck.recheck`). Ingestion hands graphiti `recall_ingest.previous_episode_uuids`. Code with its own Cypher uses the predicates there: `live_fact_predicate` for facts (the settings and admin fact lists, the dream's demotion writers), `forgotten_facts_clause` with `recallable_episode_predicate` for episodes (the dream gather and its session filter, the forget's redaction).
  - A fact is live while `expired_at` and `forgotten_at` are unset and `status` is `active`, `tentative` or unset. It is forgotten once a forget has stamped `forgotten_at`. The marker is written only by a forget (`recall_forget.py` and the cascade it runs on what the dream derived from the fact, `recall_cascade.py`; the legacy-forget backfill and the derivation backfill's cascade for older forgets), and no ingestion, dream demotion or other status change can make a forgotten fact look remembered. Older forgets are recognised too: `retracted`, an `expiration_reason` of `user_signal`, or the pre-policy shape (`expired_at` set with no `invalid_at` and no reason). `migrations/backfill_legacy_forgets.py` restamps that last shape, after which its clause can go. It restamps each graph holding the graph's write lock (below), so an ingestion running meanwhile cannot save an older copy of an edge over the restamp; a graph whose lock another writer keeps past 60 s, or that cannot be locked because Redis is unreachable, is skipped with nothing written and counted busy, a graph whose backfill raises is counted failed, and either makes the script exit 1 so it is run again. Drop the clause only after an `--apply` run that exits 0 everywhere (a dry run exits 0 too, having written nothing). An episode is recallable while it has no `redacted_at` and none of its `entity_edges` is forgotten. `recall.forgotten_fact_predicate` is never null (an unset `status` or `expiration_reason` reads as neither value), so `NOT` it is the not-forgotten test on any edge, a live one included.
  - What a forget guarantees: the fact and every episode it came from are hidden from warm context, `memory_search`, `memory_forget_search`, the settings fact list and counts, the dream's input (its episodes and facts, and every chat session a hidden episode came from, left out whole) and ingestion's extraction context (the earlier episodes graphiti shows its prompts). So is every fact the dream derived from it, and the dream text that derived it, level by level (the cascade, below). A read already under way when the forget answers does not show it either, once its last check has run: right before rendering, warm context and `memory_search` check every fact and episode they are about to show again, by uuid, in one statement (`recall_recheck.recheck`), and `memory_forget_search` its facts (`search_facts` ends with `live_now`). Every item shown passed that check, its own last graph read. A forget writes each fact's marker before anything else, so a check that began after the marker committed drops the fact and every episode citing it; a response whose check began earlier can still carry them while it is delivered (rendering, scheduling, network), and no later read will. The sentence also leaves what graphiti keeps for its own prompts (`recall_hide.py`). The edge's `fact` and `name` read `[forgotten]` (`recall.FORGOTTEN_FACT`), the originals moved to `fact_redacted` / `name_redacted`; no edge type has that name, so graphiti never runs its attribute prompt, which lists an edge's stored properties, on a forgotten edge. Every entity the fact joins, and every entity an episode citing it mentions, loses its `summary` and every property but its identity (uuid, name, group_id, labels, created_at, name_embedding): typed attributes such as `Person.role` and any `attributes` map go. The `summary` of each community one of them belongs to is blanked too. They are blanked, not rewritten: graphiti extracts them again from the next episode that mentions the entity, and the weekly rebuild restores the communities.
  - What it does not: admin audit surfaces (status `any`, the graph view) keep showing retracted facts, with the text from `fact_redacted`, and redacted episodes, on purpose, and show a tombstone's title and `hard_deleted_at` (never its content). Not scrubbed: community names, the fact's embedding vector, and a redacted episode's name and source description. A hard forget keeps, redacted, the text of an episode another edge still cites, and a tombstone keeps its name (for a stored memory, the model's short title). Two windows have no bound. When Redis cannot be reached, forgets and ingestion go ahead without the write lock (below); and a holder that loses its lease (its renewals fail, or its event loop stalls past the five-minute expiry) keeps writing while another writer takes the lock, since the renewal only warns and nothing fences the graph write. Either way an ingestion that read a fact before a forget of it can save its older copy over the forget, and that fact then stays live until a later forget of it: Redis coming back repairs nothing. And order is by when an episode is written, not when it was said, by design: an episode still queued when a forget lands is written after it, and if it states the fact again it teaches it again, as a message sent after the forget would. The weekly community rebuild does not take the lock either, so one running while a forget lands can write a community summary from entity summaries the forget then blanks. Nor does the settings page's erase-all, which deletes the whole graph: an ingestion running meanwhile can save into the emptied graph what it read before the erase.
  - Every forget goes through `recall_forget.retract`, which holds the graph's write lock (`scope_lock.py`) for the whole forget: if an ingestion holds it for all of `FORGET_LOCK_WAIT_SECONDS` (20 s), every uuid fails as `busy` and nothing is written; the chat tool waits five seconds and tries once more, and the settings page answers 409. Soft stamps `forgotten_at`, sets `status='retracted'` and `expiration_reason`, keeps an earlier `expired_at`, leaves `invalid_at` alone, then scrubs the sentence, entities and communities and stamps `redacted_at` on every episode citing the fact (`recall_hide.py`); then, still under the lock, it retracts what the dream derived from the forgotten facts (`recall_cascade.py`, below). An audit copy, once written, is never replaced by the placeholder. Hard does all of that first, the cascade too (softly, below), then (`recall_orphans.py`) turns each episode no other edge of any status cites into a tombstone (`content` emptied, `hard_deleted_at` and `redacted_at` set, its name, source description and envelope `provenance` kept, so the dream still leaves its chat session out), and last, in one query per edge, deletes the edge, drops the tombstones' mentions and deletes each entity left with no fact and no mention. Every step before the delete is idempotent and the delete finds the edge again, so repeating a forget whose clean-up failed finishes it. A step that fails is reported per edge as `cleanup_error`; recall hides the text regardless. A tombstone is never recalled, counted on the settings page or used as extraction context.
  - While Redis answers and the holder's lease holds, a forget and an ingestion never overlap. graphiti's `add_episode` saves the edges and entities it read when it began, so a forget landing mid-episode would be written over. The ingestion worker therefore holds the graph's write lock around `add_episode` and the dream metadata stamp after it (`ingest._write_locked`), waiting up to `INGEST_LOCK_WAIT_SECONDS` (60 s) for the writer holding it (a forget, or an ingestion in another process) to finish; if it cannot, the episode goes to the back of the queue once, and is dropped with a warning the second time. The lock (`scope_lock.py`) is one Redis key per graph (`MemoryScope.redis_key("write_lock")`), set NX with a token and a five-minute expiry that the holder renews, and released by the dream lock's compare-and-delete script, so a holder whose key expired never frees a newer holder's. When Redis cannot be reached both writers go ahead without it and log a warning (the unbounded windows above). graphiti also resolves each new statement against every edge between the same entities, forgotten ones included: if its model named a forgotten edge a duplicate, graphiti would append the episode to it and clear every attribute on it, the forget's marker and audit copies among them, and a contradiction would stamp its `invalid_at`. So every graphiti client is built with `recall_ingest.ForgetAwareLLMClient` (`client._build_graphiti`), which keeps a forgotten edge (its text is `[forgotten]`) out of graphiti's edge-dedup answers: a statement said again after a forget is saved by graphiti as a new live edge in the same `add_episode`, one merged into a live fact stays merged, and graphiti's model-decided dedup never rewrites a forgotten edge. There is no second extraction and nothing to repair afterwards. One path runs before the model and so past the guard: graphiti's exact-text match reuses an edge whose text equals the new statement's (lower-cased, whitespace collapsed), so a statement that reads `[forgotten]` appends its episode's uuid to the forgotten edge's `episodes`. The edge keeps its marker and audit copies and stays out of recall, and the episode, citing it, is hidden. The guard reads graphiti's prompt, so it fails closed: a candidate printed with the placeholder is forgotten wherever it appears, in the two candidate lists or not, and when the prompt shows the placeholder, anything short of both lists, once each and readable, makes the answer name no edge at all, so the statement is saved as new (at worst a duplicate of a live fact). A prompt whose lists it cannot read is logged at error once per process for each shape, so a graphiti upgrade that changes the prompt shows up before a forgotten edge meets it.
  - The dream keeps its own Cypher. Its gather and session filter (`dream/fetch.py`, `dream/hidden_sessions.py`) build on those predicates, and so does its fact gather (`live_fact_predicate`). Its writers only write over live facts: its demotions (`supersede_unless_recalled`) and single-hop neighbour invalidation (`invalidate_entity_direct_neighbors`), both in `guarded_writes.py`, match `live_fact_predicate`, and ratification (`dream/ratification.py`) supersedes with `supersede_unless_recalled` and `expected_status='tentative'` and promotes only a still-tentative, unexpired, unforgotten edge, so none of them can overwrite a forget, and none writes a new fact. Its consolidated facts and proposals are new statements, written minutes or hours after its read, so each goes to the ingestion worker with what it cites (`recall_citations.Citations`). Every one must cite the facts and episodes it was drawn from: apply keeps a citation only when it names something the pass read, filed under the kind the pass read it as, and drops a write left citing nothing before it is queued, counting it in `uncited_writes_dropped` (`dream/citations.py`; through the apply stats, `DreamPassResult`, the DreamPass record and the job status). Under the write lock, right before `add_episode`, a write resting on a fact forgotten or deleted, or on an episode no longer recallable, since the pass read the graph is dropped unwritten, logged at info, and counted in the pass's `dropped_forgotten` (it counts the drops the worker made before apply returned, which is all of them on a drained pass; the batch path does not wait, so its count covers only the drops made before it returned). The check trusts the model's citations: a write that cites one source but restates another, forgotten, in other words is not recognised. (The check still compares the statement of a write citing nothing with the sentences forgotten facts keep, but apply no longer sends one.)
  - A forget reaches what the dream derived from what it forgot. Right after the ingestion worker writes a dream write, under the same lock, it records the write's citations as `derived_from_facts` / `derived_from_episodes` (`recall_derivation.py`): on the dream's episode, which graphiti never saves again, so it is the lasting record and marks the episode as the dream's; and on every fact the write produced or merged into whose source episodes (`episodes`) all carry a record, as the union of theirs. A fact the write alone produced gets the write's citations; a user's fact a dream write merged into gets none, since it has a source no forget of the dream's citations reaches. graphiti rewrites a fact's attributes whenever its model merges a statement into it, so a later dream write merging into a derived fact rebuilds the union from the episodes' records, and a user's statement merging in leaves none. After a forget has retracted its facts and hidden their episodes, still under the lock, `recall_cascade.cascade` retracts every live fact whose record names a forgotten fact or a hidden episode and that no user's episode states (every episode it names carries a record), hides every dream episode whose record does (redacted, the entities it mentions blanked), then repeats over what that retracted and hid until a round finds nothing new. Any one forgotten source is enough. A derived fact no longer live (superseded, contradicted, forgotten already) is left as it is but walked through; one a user's own episode also states is passed over with what rests on it. A derived fact is retracted as a soft forget retracts one (`forgotten_at`, `status='retracted'`, its sentence moved to its audit copy, its entities and communities blanked, the episodes citing it redacted), with the `expiration_reason` `derived_from_forgotten:<uuid>` naming the forgotten fact it descends from. `ForgetResult.derived` lists them, the chat tool's reply counts them (`derived_uuids`) and the settings route returns `derived_forgotten`. A hard forget cascades too, softly: the derived facts are the assistant's inferences, not text the user asked to erase, retraction takes them out of every read, and deleting them would purge whatever the model's citations name, which can be more than a fact rests on, with no way back. One forget retires at most 500 derived facts and dream episodes (`CASCADE_MAX_ITEMS`) over at most 10 rounds (`CASCADE_MAX_ROUNDS`); one stopped by a bound or a failed step reports `cleanup_error` on each fact it forgot, and forgetting them again resumes from the facts already retracted for them (their reason names them). Retracted derived facts fail the live-fact test like any forgotten fact, a dream write citing one (or a hidden dream episode) is dropped by `recall_citations`, and the recall stamp never stamps them. Not reached: a citation the model left out; a dream write merged into a user's fact (its dream text is still hidden); a user's fact a derived fact's contradiction expired (graphiti's invalidation is not undone); after a hard forget, the derived facts' audit copies; and dream writes made before the records until the backfill has run. `poetry run python -m backend.copilot.graphiti.migrations.backfill_derivations` records those from each dream episode's `source_description` (at most five uuids of each kind: a consolidation listed its episodes, a proposal its facts) and reports the dream facts it could not attribute. It is a dry run unless `--apply`, which writes each graph under its write lock in batches of 500 and exits 1 while any graph was busy or failed; `--cascade-existing-forgets` also runs the cascade from every fact already forgotten.
  - A recall leaves usage on the facts it returned (`recall_stamp.py`): warm context (`dream.ratification.try_ratify_on_hit`) and `memory_search` (`recall_stamp.record_recall`) stamp each live fact they returned, one batched write per call, with `recall_count`, `last_recalled_at` and `prev_recalled_at`; a recall within 24 hours of the last one is the same use and is not counted. A forgotten or expired fact is never stamped. The stamp takes no write lock: it writes only those three properties, never the envelope, so it cannot undo a forget, and an ingestion that saves an older copy of the edge over it loses one recall and nothing else. Recall history is only ever evidence that a memory is relied on. Each of the dream's destructive writes tests it in its own statement (`recall_stamp.spared_by_recall`): a live fact last recalled within `Config.dream_demotion_protect_days` (30 by default; 0 turns this guard off) is left alone unless the reason is the user's retraction or a contradiction citing another fact the pass read, and the statement reports what it changed and what it spared, in one row however many facts it reached (`guarded_writes.py`). `dream/demotions.py` counts in `protected_demotions` the distinct facts an acknowledged write spared that one read, made after every acknowledged write, finds live. The count is a snapshot at that read: a later forget or pass can retire a counted fact. A write that raised may have committed, never arrived, or still be queued on FalkorDB (the read, a `GRAPH.RO_QUERY`, does not wait for queued writes), so it is counted in `indeterminate_demotion_writes` rather than as a failure, and `demotion_accounting_complete` is True only when the read answered and no write's outcome is unknown; otherwise the count is provisional. Ratification follows the same rule: a promotion or supersession whose write raised leaves the sweep's counts and the warm-context hit hook's count provisional (`RatificationResult.accounting_complete`, `HitRatification.accounting_complete`), while a per-edge error whose outcome is known does not. Entity invalidation has no degree cap: a hub is invalidated whole, behind the `dream-pass-invalidate-entity` flag, as before recall stamps. The ratification sweep's supersession of an unratified proposal carries the same guard with no override, so a proposal recalled within the window stays tentative even when its Redis hit count was lost (counted in `RatificationResult.protected_count`). Nothing is read beforehand to decide, and usage never changes which operations a pass attempts, so a pass never changes more facts with usage data than without. The sanitize prompt separately asks the model not to demote a recalled fact for staleness at all. No recalls means nothing, and retrieval order ignores the stamps. An edge written before the stamps reads as never recalled; nothing is backfilled.
  - A live fact whose valid time has ended (a past `invalid_at`, never expired) is still recalled, as history: it renders as `fact (valid: X — Y)`. Only a retired fact gets a retirement label, and "present" appears only while no end is known.

## Dream Schedules

A memory scope (the account, or one hired expert) that has been registered has one `MemoryScopeSchedule` row, and while that row is ACTIVE, the crons whose flag is on: `dream_nightly_batch_{scope_key}` (daily, 03:00 owner-local) and `community_rebuild_{scope_key}` (weekly, 04:00 owner-local on Mondays: APScheduler reads the `0` in `0 4 * * 0` as Monday). The account's scope key is its user id, so its job ids predate the table. The table, not the Redis markers (written for diagnostics only), decides whether a scope is scheduled; `copilot/dream/registry.py` keeps it in step:

- **Registration.** Ingest registers a scope when its group's queue is created: the group's first write in a process, and again whenever the queue comes back after retiring (60 s idle), with at most one registration in flight per group. A hire or raise registers the expert in the background. Registration never reactivates a PAUSED or WIPED scope, and writes nothing while both of the owner's dream flags are off, so a scope that was never registered or paused has no row.
- **Lifecycle.** Archive and a schedule pause (including a budget breach) pause the expert's scope and remove its crons; the resume route and a revive resume a PAUSED scope (or register one that has no row). A lifecycle change the deadline cuts off is not retried: the cron gate still keeps an archived or paused expert's crons from running, and a later lifecycle change or the backfill settles the row (an ordinary registration does not look at the expert's lifecycle). `mark_wiped` is a seam for the wipe work: nothing calls it yet and it does not erase the graph. No registry operation takes a scope out of WIPED: registration and resume skip it, and pause or archive leave it WIPED (a pause only moves an ACTIVE or PAUSED row, checked in the same statement), so the wipe work decides what brings one back.
- **Timezone.** A timezone change re-registers every scope of the user in the new zone. If that fails, the next registration picks the new zone up (a returning queue, a lifecycle change, the backfill), not necessarily the next write while the group's worker is alive.
- **Deadlines.** Every registry call to the DatabaseManager, the scheduler, the flag backend or the user lookup runs under a 10 s deadline (`copilot/dream/deadline.py`), the Redis markers under their own 5 s one, all built on `asyncio.timeout` so a deadline nested in another still holds on Python 3.11. On timeout the call's cancellation is requested; a timeout does not prove the call had no effect, since the server may already have acted on it. The API hooks bound the whole change, so the memory part of a request ends when that deadline requests cancellation, plus the cancelled call's cleanup and any time the event loop itself is blocked, and a cron whose registry gate cannot answer in time skips that run. Background registration (ingest, hire, raise) and the backfill pay the deadline per call, so during an outage the backfill's run time grows with the number of scopes.

Existing deployments fill the table with `poetry run memory-schedule-backfill` (`--dry-run` only counts; `--force` re-registers every cron, e.g. after jobs were lost from the scheduler). It schedules every account with a `user_*` FalkorDB graph and a user row, resumes or registers every live hired expert, pauses paused and archived ones, and prints a JSON report that lists each failed scope with its cron and reason; it exits 1 if anything failed. Re-running is safe; it registers nothing twice, but pauses the paused experts again (a state write and a remove call each). It needs Postgres, Redis, FalkorDB, the scheduler service and the flag backend, so run it where the backend services' environment is set; crons are only registered for owners with `dream-pass-enabled` or `graphiti-communities-enabled` on.

**Deploy order.** Apply the migration and deploy the DatabaseManager and scheduler services first, then the writers (REST API, copilot-executor), then run the backfill: newer writers get a 404 from an older scheduler's scope RPCs, and a newer scheduler's cron gate skips every memory run, the account's included, while an older DatabaseManager cannot answer the registry. Expert crons pass `expert_id`, which an older scheduler's cron bodies do not accept, so they are not rollback-compatible; account crons are.

## Query Cookbook

Run everything from `autogpt_platform/backend` and use `poetry run ...`.

Get the `group_id` for a user (use `MemoryScope.for_expert(user_id, expert_id)`
for one of their experts; the snippets below take either):

```bash
poetry run python - <<'PY'
from backend.copilot.graphiti.scope import MemoryScope
print(MemoryScope.for_user("883cc9da-fe37-4863-839b-acba022bf3ef").group_id)
PY
```

Inspect graph counts:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
SCOPE = MemoryScope.for_user(USER_ID)

QUERIES = {
    "entities": "MATCH (n:Entity) RETURN count(n) AS count",
    "episodes": "MATCH (n:Episodic) RETURN count(n) AS count",
    "communities": "MATCH (n:Community) RETURN count(n) AS count",
    "relates_to_edges": "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS count",
}

async def run():
    driver = open_driver(SCOPE)
    try:
        for name, query in QUERIES.items():
            records, _, _ = await driver.execute_query(query)
            print(name, records[0]["count"])
    finally:
        await driver.close()

asyncio.run(run())
PY
```

List entities or relation-name counts:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
SCOPE = MemoryScope.for_user(USER_ID)

async def run():
    driver = open_driver(SCOPE)
    try:
        records, _, _ = await driver.execute_query(
            "MATCH (n:Entity) RETURN n.name AS name, n.summary AS summary ORDER BY n.name"
        )
        print("## entities")
        for row in records:
            print(row)

        records, _, _ = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            RETURN e.name AS relation, count(e) AS count
            ORDER BY count DESC, relation
            """
        )
        print("\\n## relation_counts")
        for row in records:
            print(row)
    finally:
        await driver.close()

asyncio.run(run())
PY
```

Inspect facts around one node:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
SCOPE = MemoryScope.for_user(USER_ID)
TARGET = "sarah"

async def run():
    driver = open_driver(SCOPE)
    try:
        records, _, _ = await driver.execute_query(
            """
            MATCH (a)-[e:RELATES_TO]->(b)
            WHERE (exists(a.name) AND toLower(a.name) = $target)
               OR (exists(b.name) AND toLower(b.name) = $target)
            RETURN a.name AS source, e.name AS relation, e.fact AS fact, b.name AS target
            ORDER BY e.created_at
            """,
            target=TARGET,
        )
        for row in records:
            print(row)
    finally:
        await driver.close()

asyncio.run(run())
PY
```

Inspect all chat messages for a user:

```bash
poetry run python - <<'PY'
import asyncio
from prisma import Prisma

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"

async def run():
    db = Prisma()
    await db.connect()
    try:
        rows = await db.query_raw(
            '''
            select cm."sessionId" as session_id,
                   cm.sequence,
                   cm.role,
                   left(cm.content, 260) as content,
                   cm."createdAt" as created_at
            from "ChatMessage" cm
            join "ChatSession" cs on cs.id = cm."sessionId"
            where cs."userId" = $1
            order by cm."createdAt", cm.sequence
            ''',
            USER_ID,
        )
        for row in rows:
            print(row)
    finally:
        await db.disconnect()

asyncio.run(run())
PY
```

Notes:

- `RELATES_TO` edges hold semantic facts. Inspect `e.name` and `e.fact`.
- `MENTIONS` edges are provenance from episodes to extracted nodes.
- Prefer directed queries `->` when checking for duplicates; undirected matches double-count mirrored edges.
