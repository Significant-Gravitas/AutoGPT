# Open Source Contribution — AutoGPT

## Project

[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) — a platform for building, deploying, and running continuous AI agents, including an agent marketplace and library.

**Forked to:** [Kidus5168/AutoGPT](https://github.com/Kidus5168/AutoGPT)
**Working branch:** [`fix-issue-9879`](https://github.com/Kidus5168/AutoGPT/tree/fix-issue-9879) (this write-up) / [`fix/9879-carry-marketplace-image`](https://github.com/Kidus5168/AutoGPT/tree/fix/9879-carry-marketplace-image) (the implementation, submitted as [PR #13639](https://github.com/Significant-Gravitas/AutoGPT/pull/13639))

## Selected Issue

**[#9879 — Marketplace data should be downloaded with Agent](https://github.com/Significant-Gravitas/AutoGPT/issues/9879)** (filed 2025-04-25)

## Problem Summary (Phase I)

When a user downloads an agent from the marketplace into their personal library, the agent shows up with stale information rather than the details it was published with:

1. **Wrong title.** The downloaded agent displays the creator's *original* graph title instead of the title the agent was published under on the marketplace.
2. **Missing images.** The agent's marketplace image is not carried over, so the library entry has no artwork.

**Expected behavior:** A downloaded agent should appear in the user's library exactly as it was presented on the marketplace — with the marketplace title and image intact.

## Environment Setup

**Approach used:** README instructions + static code inspection, rather than a full running stack.

The `autogpt_platform` backend's documented setup (`autogpt_platform/README.md`) is Docker-based:
```
cp .env.default .env
docker compose up -d       # or: make start-core / make run-backend
```
It requires Docker, a generated Prisma client, and a Postgres instance via Supabase.

**Real challenge encountered:** I didn't have Docker/Postgres available in my working environment, so I couldn't bring up the full stack or run `pytest` against a live database. I worked around this by:
- Shallow-cloning the repo and reading the source directly (`backend/api/features/library/_add_to_library.py`, `model.py`) instead of relying on a running server to observe behavior.
- Reading `schema.prisma` directly to confirm the DB columns involved (`LibraryAgent.imageUrl`, `StoreListingVersion.name`/`imageUrls`) rather than inspecting a live database.
- Verifying the eventual code change with `python -m py_compile` (syntax/parse check) instead of a full test run, and writing new unit tests that follow the existing mock-based patterns in `_add_to_library_test.py` (which don't require a real DB) so they can run under the project's CI.

## Codebase Analysis (Phase II) — UMPIRE

### Understand

Marketplace listings live on `StoreListingVersion` (`autogpt_platform/backend/schema.prisma`), which has its own `name`, `description`, and `imageUrls: String[]` — independent of the underlying `AgentGraph`'s own `name`/`description`. When an agent is downloaded, the code that creates the user's `LibraryAgent` row never reads those marketplace-specific fields, so the library entry ends up describing the creator's private graph instead of the public listing.

### Match

Draft PR [#11347](https://github.com/Significant-Gravitas/AutoGPT/pull/11347) (open since Nov 2025, stale/conflicted) attempted the *title* half of this exact fix: adding nullable `name`/`description` override columns to `LibraryAgent`, populated from the marketplace listing, with `from_db()` falling back to the graph's values when unset. It's a directly analogous precedent — but it never touched the *image* half of the issue, and stalled before merging.

I also dated the bug: the `LibraryAgent.imageUrl` column was added by migration `20250203133647_add_image_url` (2025-02-03) — **before** this issue was even filed (2025-04-25). The database was already prepared to store a per-agent image; the download code path simply never wrote to it.

### Plan

Root cause, with exact files/functions:

- `autogpt_platform/backend/backend/api/features/library/_add_to_library.py`
  - `resolve_graph_for_library()` fetches the `StoreListingVersion` (`slv`) — its `name`/`imageUrls` are in hand — but returns only `slv.AgentGraph` as a `GraphModel`, discarding them.
  - `add_graph_to_library()` builds the new `LibraryAgent` from the graph only. Its `LibraryAgent.prisma().create(...)` call connects the `AgentGraph` but never sets `imageUrl`, and there's no column to persist a marketplace title at all.
- `autogpt_platform/backend/backend/api/features/library/model.py` — `LibraryAgent.from_db()`
  - `name=graph.name` (always the creator's private graph title, never the listing's).
  - `image_url=agent.imageUrl` (null, because it's never populated on download).

Two-part fix, split because only one part needs a schema migration:

1. **Image (no migration needed — the column already exists):** in `add_graph_to_library()`, look up `StoreListingVersion.imageUrls` for the listing being downloaded and set `LibraryAgent.imageUrl` from the first entry, on both the create path and the soft-delete restore path (guarding the restore so it never blanks out an image the user already has).
2. **Title (needs a migration):** add a nullable title/description override column to `LibraryAgent` (following PR #11347's approach, rebased), populated at download time, with `from_db()` preferring it over `graph.name` when present.

### Review

- Self-created agents (`isCreatedByUser=True`) don't go through `add_graph_to_library`'s marketplace path, so the fix can't regress how a user's own agents are named/imaged.
- The restore path (`UniqueViolationError` → `update()`) must not blank an existing image when the listing happens to have none — handled by only setting `imageUrl` when a marketplace image is actually found.
- `add_store_agent_to_library_as_admin` shares the same helper, so the fix covers the admin-download path too without extra changes.

### Evaluate

- Added `test_add_graph_to_library_carries_marketplace_image` (image is written to `imageUrl` on create) and `test_fetch_marketplace_image_url_returns_first_image` (covers first-image / empty-list / no-listing cases) to `_add_to_library_test.py`, plus updated the two pre-existing tests to mock the new lookup.
- Verified with `python -m py_compile` on both edited files (no live DB available, see Environment Setup above).
- Did not run the full `pytest` suite locally — requires a generated Prisma client + Postgres. The new tests follow the same mock-based pattern as the existing suite, so they should run under CI.

## Implementation (Phase III)

**Pull Request:** [Significant-Gravitas/AutoGPT#13639](https://github.com/Significant-Gravitas/AutoGPT/pull/13639)
**Branch:** [`fix/9879-carry-marketplace-image`](https://github.com/Kidus5168/AutoGPT/tree/fix/9879-carry-marketplace-image) (on my fork)

I implemented the **image carry-over** half of the plan above — the part that doesn't need a database migration:

- Added a helper `_fetch_marketplace_image_url()` that reads the first image from the `StoreListingVersion.imageUrls` for the listing being downloaded.
- Set `LibraryAgent.imageUrl` from that value on the **create** path, and on the **restore** path for a previously soft-deleted entry.

**Files changed:**
- `autogpt_platform/backend/backend/api/features/library/_add_to_library.py` — the fix
- `autogpt_platform/backend/backend/api/features/library/_add_to_library_test.py` — tests

### Known limitation / next step

The **wrong-title** half of the issue is *not* fixed on this branch. `LibraryAgent` has no title/name column — the displayed name is derived from the graph in `LibraryAgent.from_db()`. A correct, persistent title fix needs a new `LibraryAgent` column plus a Prisma migration (see Plan step 2 above), which I've scoped as the follow-up commit.

## Status

- Phase II plan above, with root cause, an analogous precedent (PR #11347), and a two-part UMPIRE plan.
- Phase III — image carry-over implemented, tested (unit + syntax), and submitted as [PR #13639](https://github.com/Significant-Gravitas/AutoGPT/pull/13639) to the upstream repo. [Interest comment posted on the issue](https://github.com/Significant-Gravitas/AutoGPT/issues/9879#issuecomment-4954807673).
- Title fix (needs a schema migration) is the documented next step.
