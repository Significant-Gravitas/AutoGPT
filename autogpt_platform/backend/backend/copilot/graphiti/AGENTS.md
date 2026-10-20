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
  - A fact is live while `expired_at` and `forgotten_at` are unset and `status` is `active`, `tentative` or unset. It is forgotten once a forget has stamped `forgotten_at`. The marker is written only by a forget (`recall_forget.py`, and the legacy-forget backfill for older forgets), and no ingestion, dream demotion or other status change can make a forgotten fact look remembered, nor a remembered one look forgotten: a `status` of `retracted` or an `expiration_reason` of `user_signal` is no forget without the marker, since graphiti's attribute extraction fills both from an episode's text and a dream demotion picks its reason. The extraction schema (`types.MemoryFact`) leaves both out, and an extracted `retracted` is put back to `active` as the episode is written (`ingest._undo_extracted_retractions`). Nor can extraction write the marker: graphiti-core 0.30.2 saves every key its attribute model answers, so `recall_ingest.ForgetAwareLLMClient` keeps only what the edge or entity type declares (`types.extracted_attributes`), and no extraction model declares the marker or the audit copies (`types.POLICY_EDGE_FIELDS`). Forgets made before the marker are recognised by the pre-policy shape (`expired_at` set with no `invalid_at` and no reason). `migrations/backfill_legacy_forgets.py` restamps that shape, after which its clause can go. It restamps each graph holding the graph's write lock (below), so an ingestion running meanwhile cannot save an older copy of an edge over the restamp; a graph whose lock another writer keeps past 60 s, or that cannot be locked because Redis is unreachable, is skipped with nothing written and counted busy, a graph whose backfill raises is counted failed, and either makes the script exit 1 so it is run again. Drop the clause only after an `--apply` run that exits 0 everywhere (a dry run exits 0 too, having written nothing). An episode is recallable while it has no `redacted_at` and none of its `entity_edges` is forgotten.
  - What a forget guarantees: the fact and every episode it came from are hidden from warm context, `memory_search`, `memory_forget_search`, the settings fact list and counts, the dream's input (its episodes and facts, and every chat session a hidden episode came from, left out whole) and ingestion's extraction context (the earlier episodes graphiti shows its prompts). A read already under way when the forget answers does not show it either, once its last check has run: right before rendering, warm context and `memory_search` check every fact and episode they are about to show again, by uuid, in one statement (`recall_recheck.recheck`), and `memory_forget_search` its facts (`search_facts` ends with `live_now`). Every item shown passed that check, its own last graph read. A forget writes each fact's marker before anything else, so a check that began after the marker committed drops the fact and every episode citing it; a response whose check began earlier can still carry them while it is delivered (rendering, scheduling, network), and no later read will. The sentence also leaves what graphiti keeps for its own prompts (`recall_hide.py`). The edge's `fact` and `name` read `[forgotten]` (`recall.FORGOTTEN_FACT`), the originals moved to `fact_redacted` / `name_redacted`; no edge type has that name, so graphiti never runs its attribute prompt, which lists an edge's stored properties, on a forgotten edge. Every entity the fact joins, and every entity an episode citing it mentions, loses its `summary` and every property but its identity (uuid, name, group_id, labels, created_at, name_embedding): typed attributes such as `Person.role` and any `attributes` map go. The `summary` of each community one of them belongs to is blanked too. They are blanked, not rewritten: graphiti extracts them again from the next episode that mentions the entity, and the weekly rebuild restores the communities.
  - What it does not: admin audit surfaces (status `any`, the graph view) keep showing retracted facts, with the text from `fact_redacted`, and redacted episodes, on purpose, and show a tombstone's title and `hard_deleted_at` (never its content). `memory_forget_search` lists forgotten facts too when the chat asks for them (`include_forgotten`), so a soft forget can be made permanent with a hard one, but never what they said: each is `[forgotten]` with its uuid and when it was stored and forgotten (each read as a timestamp and written out anew, else `unknown`), without its text, audit copy, scope or valid dates. `recall.search_forgotten_facts` finds them by the embedding a forget keeps, ranking the forgotten facts alone, so no live fact takes their slots. Not scrubbed: community names, the fact's embedding vector, and a redacted episode's name and source description. A hard forget keeps, redacted, the text of an episode another edge still cites, and a tombstone keeps its name (for a stored memory, the model's short title). Two windows have no bound. When Redis cannot be reached, forgets and ingestion go ahead without the write lock (below); and a holder that loses its lease (its renewals fail, or its event loop stalls past the five-minute expiry) keeps writing while another writer takes the lock, since the renewal only warns and nothing fences the graph write. Either way an ingestion that read a fact before a forget of it can save its older copy over the forget, and that fact then stays live until a later forget of it: Redis coming back repairs nothing. And order is by when an episode is written, not when it was said, by design: an episode still queued when a forget lands is written after it, and if it states the fact again it teaches it again, as a message sent after the forget would. The weekly community rebuild does not take the lock either, so one running while a forget lands can write a community summary from entity summaries the forget then blanks. Nor does the settings page's erase-all, which deletes the whole graph: an ingestion running meanwhile can save into the emptied graph what it read before the erase.
  - Every forget goes through `recall_forget.retract`, which holds the graph's write lock (`scope_lock.py`) for the whole forget: if an ingestion holds it for all of `FORGET_LOCK_WAIT_SECONDS` (20 s), every uuid fails as `busy` and nothing is written; the chat tool waits five seconds and tries once more, and the settings page answers 409. Soft stamps `forgotten_at`, sets `status='retracted'` and `expiration_reason`, keeps an earlier `expired_at`, leaves `invalid_at` alone, then scrubs the sentence, entities and communities and stamps `redacted_at` on every episode citing the fact (`recall_hide.py`). An audit copy, once written, is never replaced by the placeholder. Hard does all of that first, then (`recall_orphans.py`) turns each episode no other edge of any status cites into a tombstone (`content` emptied, `hard_deleted_at` and `redacted_at` set, its name, source description and envelope `provenance` kept, so the dream still leaves its chat session out), and last, in one query per edge, deletes the edge, drops the tombstones' mentions and deletes each entity left with no fact and no mention. Every step before the delete is idempotent and the delete finds the edge again, so repeating a forget whose clean-up failed finishes it. A step that fails is reported per edge as `cleanup_error`; recall hides the text regardless. A tombstone is never recalled, counted on the settings page or used as extraction context.
  - While Redis answers and the holder's lease holds, a forget and an ingestion never overlap. graphiti's `add_episode` saves the edges and entities it read when it began, so a forget landing mid-episode would be written over. The ingestion worker therefore holds the graph's write lock around `add_episode` and the dream metadata stamp after it (`ingest._write_locked`), waiting up to `INGEST_LOCK_WAIT_SECONDS` (60 s) for the writer holding it (a forget, or an ingestion in another process) to finish; if it cannot, the episode goes to the back of the queue after a pause (one second, doubling with each wait up to 30 s) and waits again, as often as it takes: contention never drops an episode while the process runs, and every third wait a warning names the episode still unwritten. The queue lives in the process: what it holds when the process stops is lost. Waiters take turns: a writer that finds the lock held claims the next turn (the lock's key plus `:next`, renewed at every poll, lapsing two seconds after its waiter stops), and while it holds that claim nobody else takes the lock, the writer that just let it go included, so a worker writing a backlog hands the lock over between two episodes. The lock (`scope_lock.py`) is one Redis key per graph (`MemoryScope.redis_key("write_lock")`), set NX with a token and a five-minute expiry that the holder renews, and released by the dream lock's compare-and-delete script, so a holder whose key expired never frees a newer holder's. When Redis cannot be reached both writers go ahead without it and log a warning (the unbounded windows above). graphiti also resolves each new statement against every edge between the same entities, forgotten ones included: if its model named a forgotten edge a duplicate, graphiti would append the episode to it and clear every attribute on it, the forget's marker and audit copies among them, and a contradiction would stamp its `invalid_at`. So every graphiti client is built with `recall_ingest.ForgetAwareLLMClient` (`client._build_graphiti`), which keeps a forgotten edge (its text is `[forgotten]`) out of graphiti's edge-dedup answers: a statement said again after a forget is saved by graphiti as a new live edge in the same `add_episode`, one merged into a live fact stays merged, and graphiti's model-decided dedup never rewrites a forgotten edge. There is no second extraction and nothing to repair afterwards. One path runs before the model and so past the guard: graphiti's exact-text match reuses an edge whose text equals the new statement's (lower-cased, whitespace collapsed), so a statement that reads `[forgotten]` appends its episode's uuid to the forgotten edge's `episodes`. The edge keeps its marker and audit copies and stays out of recall, and the episode, citing it, is hidden. The guard reads graphiti's prompt, so it fails closed: a candidate printed with the placeholder is forgotten wherever it appears, in the two candidate lists or not, and when the prompt shows the placeholder, anything short of both lists, once each and readable, makes the answer name no edge at all, so the statement is saved as new (at worst a duplicate of a live fact). A prompt whose lists it cannot read is logged at error once per process for each shape, so a graphiti upgrade that changes the prompt shows up before a forgotten edge meets it.
  - The dream keeps its own Cypher. Its gather and session filter (`dream/fetch.py`, `dream/hidden_sessions.py`) build on those predicates, and so does its fact gather (`live_fact_predicate`). Its writers only write over live facts: supersession (`mark_edges_superseded`) and single-hop neighbour invalidation (`invalidate_entity_direct_neighbors`) match `live_fact_predicate` unless a caller passes `expected_status`, and ratification (`dream/ratification.py`) passes `expected_status='tentative'` and promotes only a still-tentative, unexpired, unforgotten edge, so none of them can overwrite a forget, and none writes a new fact. Its consolidated facts and proposals are new statements, written minutes or hours after its read, so each goes to the ingestion worker with what it cites (`recall_citations.Citations`; one citing nothing is taken to rest on everything the pass read, and its statement is compared, lower-cased and whitespace-collapsed, with the sentences forgotten facts keep). Under the write lock, right before `add_episode`, a write resting on a fact forgotten or deleted, or on an episode no longer recallable, since the pass read the graph is dropped unwritten, logged at info, and counted in the pass's `dropped_forgotten` (it counts the drops the worker made before apply returned, which is all of them on a drained pass; the batch path does not wait, so its count covers only the drops made before it returned). The pass counts a write consolidated, or a proposal proposed, only once graphiti has saved it: those still queued or being written when apply returns are reported as `pending_writes` (every one on the batch path), and those whose write raised as `failed_writes`. The check trusts the model's citations: a write that cites one source but restates another, forgotten, in other words is not recognised, and an uncited write drawn only from a chat session whose episode and fact the pass did not read (beyond its 50 newest episodes and 500 facts) is recognised only by an exact restatement of a softly forgotten sentence.
  - A live fact whose valid time has ended (a past `invalid_at`, never expired) is still recalled, as history: it renders as `fact (valid: X — Y)`. Only a retired fact gets a retirement label, and "present" appears only while no end is known.

## Dream Schedules

A memory scope (the account, or one hired expert) that has been registered has one `MemoryScopeSchedule` row, and while that row is ACTIVE, weekly community maintenance and the independently flag-gated nightly dream: `dream_nightly_batch_{scope_key}` (daily, 03:00 owner-local) and `community_rebuild_{scope_key}` (weekly, 04:00 owner-local on Mondays: APScheduler reads the `0` in `0 4 * * 0` as Monday). The account's scope key is its user id, so its job ids predate the table. The table decides whether a scope is scheduled; `copilot/dream/registry.py` keeps it in step:

- **Registration.** Ingest registers a scope when its group's queue is created: the group's first write in a process, and again whenever the queue comes back after retiring (60 s idle), with at most one registration in flight per group. A hire or raise registers the expert in the background. Registration never reactivates a PAUSED or WIPED scope. Community maintenance needs no feature flag; the dream and account-only skill learning keep their independent gates.
- **Lifecycle.** Archive and a schedule pause (including a budget breach) pause the expert's scope and remove its crons; the resume route and a revive resume a PAUSED scope (or register one that has no row). A lifecycle change the deadline cuts off is not retried: the cron gate still keeps an archived or paused expert's crons from running, and a later lifecycle change or the backfill settles the row (an ordinary registration does not look at the expert's lifecycle). `mark_wiped` is a seam for the wipe work: nothing calls it yet and it does not erase the graph. No registry operation takes a scope out of WIPED: registration and resume skip it, and pause or archive leave it WIPED (a pause only moves an ACTIVE or PAUSED row, checked in the same statement), so the wipe work decides what brings one back.
- **Reconcile.** Registration, pause and resume act on events, and two of them can interleave: a pause's takedown that lands after a later resume registered the same deterministic job ids, or a registration that read the row before a timezone change and lands after it. Each can leave an ACTIVE row whose recorded crons the scheduler lacks or runs in the wrong timezone, which an ensure, trusting the recorded ids, never notices. On each scheduler start and then every 30 minutes, the scheduler runs `reconcile_memory_scope_crons` (`copilot/dream/scope_reconcile.py`). It lists its dream-system crons (inside the pass's 20-minute ceiling, the database stopping the query after 60 s) and every registry row, paged, then settles each scope out of step, on the row as read again right before. Every cron an ACTIVE row records that the scheduler lacks, or runs in another timezone, is registered in the row's timezone and tagged as the reconcile's (` (reconciled)` on the job's name). A tagged cron its row does not record, or whose row is not ACTIVE or gone, is removed, before the scope's crons are repaired and whatever that repair does: so a restore that raced a delete or a pause (the account's delete RPCs have the registry forget the cron before they remove it, and keep it, failing, when the forget fails), failed partway, was cancelled or finished after its pass, leaves at worst a tagged cron the next pass that reaches the scope takes away, even while the scope's repair keeps failing. An untagged cron the row does not record is the registry's (an account's from before the registry, a registration whose record failed) and is left alone, and no flag is read. Every cron of a PAUSED or WIPED row, or of an expert with no row, is removed; an account's untagged crons with no row are left, as the cron gate lets them run. Every change is idempotent (a removal that finds the job already gone is no error). Each of the pass's own jobstore writes gives up after 10 s on a row another transaction holds (`jobstore.WRITE_TIMEOUT`: a transaction-local `lock_timeout` and `statement_timeout`; the scheduler's other writes are not limited), failing only that scope. A pass walks the scopes as a ring from just after the last scope a pass in this process took (a random one after the process starts), taking each as its place before it works it, so a pass cut off at its ceiling, even inside a scope's work, hands the rest of the ring to the next, and a restart can't keep repairing the same first scopes. The place is the process's own: APScheduler 3 can't share a jobstore between processes, so production runs one scheduler, and during a rollout two each walk on their own. So a lost, stale or stray cron is put right once the walk next reaches its scope, not within a fixed 30 minutes. The job lives in a process-local `MemoryJobStore`, registered after start with an immediate next run; it never enters the shared SQL job table. During a rolling deploy each new scheduler runs its own pass; overlapping passes replay bounded, idempotent work. Expert crons use a separate callable reference so old binaries park them rather than execute incompatible kwargs. A new-code listing that includes a parked row restores its SQL next-run timestamp from the readable payload, guarded by unchanged id, payload and NULL column. The memory-prefix listing, schedule-index reconciliation at startup/daily, and all-jobs cache can recover rows this way. Explicitly paused payloads stay paused. An overdue recovered cron coalesces missed runs subject to its misfire grace; an old pod can park it again until the overlap ends. A registration whose record finds the row no longer ACTIVE reports `scope_changed` and takes nothing down (a takedown there could take a concurrent resume's crons): the gate holds the scope, and the reconcile removes its crons. The cron gate itself only skips a straggler's run; removing it is the reconcile's job, and so is restoring the crons a resume landing inside a pause's takedown lost.
- **Timezone.** A timezone change re-registers every scope of the user in the new zone. The new timezone is recorded before cron replacement. A failed replacement retains the existing recorded job id so reconciliation can detect the old timezone and retry; a first registration that fails still has no id. A failure to record timezone intent is retried by a later registration or backfill.
- **Deadlines.** Every registry call to the DatabaseManager, the scheduler, the flag backend or the user lookup runs under a 10 s deadline (`copilot/dream/deadline.py`), built on `asyncio.timeout` so a deadline nested in another still holds on Python 3.11. On timeout the call's cancellation is requested; a timeout does not prove the call had no effect, since the server may already have acted on it. The API hooks bound the whole change, so the memory part of a request ends when that deadline requests cancellation, plus the cancelled call's cleanup and any time the event loop itself is blocked, and a cron whose registry gate cannot answer in time skips that run. Background registration (ingest, hire, raise) and the backfill pay the deadline per call, so during an outage the backfill's run time grows with the number of scopes.

Existing deployments fill the table with `poetry run memory-schedule-backfill` (`--dry-run` only counts; `--force` re-registers every cron, e.g. after jobs were lost from the scheduler). It discovers accounts with a `user_*` FalkorDB graph and a user row, accounts with skill-learning sources, and existing ACTIVE account rows; it ensures those accounts, resumes or registers every live hired expert, pauses paused and archived ones, and prints a JSON report that lists each failed scope with its cron and reason; it exits 1 if anything failed. Re-running is safe; it registers nothing twice, but pauses the paused experts again (a state write and a remove call each). It needs Postgres, FalkorDB and the scheduler service, so run it where the backend services' environment is set. Community registration needs no flag; the dream uses `dream-pass-enabled` and account-only skill learning uses the independent `dream-skill-learning-enabled` flag. Force refresh does not override WIPED or paused account state. Dream and learning cron bodies skip work while their flag is off. Unavailable flag evaluations preserve recorded intent, including during timezone changes; reconciliation can still restore those crons.

**Deploy order.** Apply both registry migrations and deploy the DatabaseManager and scheduler services first, then the writers (REST API, copilot-executor), then run the backfill: newer writers get a 404 from an older scheduler's scope RPCs, and a newer scheduler's cron gate skips every memory run, the account's included, while an older DatabaseManager cannot answer the registry. Rolling scheduler overlap can temporarily park expert crons; the new scheduler recovers them on a later listing, including its startup/periodic reconciliation. Account cron references and kwargs remain backward-compatible. For rollback, stop or remove expert crons before reverting all schedulers: old binaries leave them parked, and no new reader remains to recover them. The process-local reconcile needs no persistent system-job repair. Backfill follows service deployment and is needed for dormant-scope adoption, not as a prerequisite for the additive schema deployment.

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
