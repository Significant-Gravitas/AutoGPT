# Inline OpenUI capabilities and drafts

OpenUI composes 40 bounded components inside the existing Copilot conversation. It has no separate sample page or environment flag. The model must retrieve or receive actual data before presenting it. Rendering a view does not execute an external action.

## Task coverage

| User task | Components and behavior |
| --- | --- |
| Inspect generated images, video, audio or files | `Artifact` uses native output renderers and authenticated workspace downloads, captions, filenames and retry. `ResultList` also supports image options. Supply actual output URIs; do not invent an image or claim that rendering generates it. |
| Explore a dashboard | `Dashboard` connects filters, aggregates, multi-series charts, a table and CSV export. Bind it to a `DataGrid` when edits should update every view. Unknown amounts remain unknown. |
| Clean and compare records | `DataGrid` provides typed cells, row selection, sorting, search, pagination, row addition/deletion and CSV. Calculated numeric columns recompute from source columns. Invalid cells block submission even when filtered offscreen. |
| Choose or prioritize | `Comparison` retains mobile A/B swipes, labeled buttons and undo. `ResultList` supports up to 100 rich alternatives with search and pagination. `RankedList` provides accessible move buttons. `SearchSelect` supports larger compact selections. |
| Supply structured preferences | `ValidatedField`, `DateRangeField`, `DateTimeField`, `RangeField` and `FieldGroup` extend existing forms with typed validation, conditional groups and repeated records. A nullable boolean distinguishes unknown from no. |
| Review a prose draft | `DocumentEditor` keeps the original beside editable current text, reset and download. It never sends email or writes an external document itself. Use typed controls for independently editable facts. |
| Plan dated work and travel | `Itinerary` connects dates, IANA timezone, availability, stable tasks, supplied locations, directed travel estimates, map selection and calendar export. `TaskBoard` tracks local owner/due/status/dependencies; it does not claim backend work completed. |
| Follow a real agent run | `ExecutionView` and the native execution card expand `InlineExecution` in chat. Real graph/run IDs fetch backend status and native outputs. Stop waits for backend confirmation; retry returns through the normal chat and approval flow. |

Existing metrics, charts, tables, maps, checklists, cost tables, calculated metrics and schedules remain available. Layout wrappers and child records count toward the catalog total; 40 is not a count of independent product features.

## Data and interaction contracts

- Every newly published `Workspace` requires a stable fourth-argument `viewId`. Reuse it and form/field/row/task/place IDs across revisions; choose another ID for another task. The parser still accepts older saved views without this field.
- Typed numeric drafts preserve invalid letters and show an accessible local warning, including during corrections. Only valid numbers reach the submitted chat message. Unknown inputs do not silently turn into zero.
- Hidden conditional groups retain edits but neither submit hidden values nor block the visible form. Submission removes internal derived fields and rejects oversized payloads without truncating user text.
- Numeric data sorts numerically, including negatives and decimals. CSV escapes quoted content and formula-leading text. Calculated columns support two-column add/subtract/multiply/divide; missing inputs, division by zero and overflow produce unknown values.
- Charts retain null gaps. Units must agree with numeric columns; stacked charts reject negative values. A shared dashboard applies its filters to charts, metrics, table and export together.
- Dates must exist, ranges must be ordered and increments must fit. Local times in daylight-saving gaps are rejected; repeated local times require an explicit occurrence. No timezone is silently inferred.
- Itinerary transfers are directional and date-specific. A changed order or date without a matching estimate pauses the plan. Completed prefix tasks establish the current location without consuming future time. Calendar events use actual UTC slots and tentative status; downloading does not create a booking or update a remote calendar. The export follows [RFC 5545](https://www.rfc-editor.org/rfc/rfc5545) escaping and line folding.
- Private workspace previews and execution fetches are withheld on public chat shares. Existing separately authorized execution-share links keep their existing behavior.

## Draft persistence and deployment

Apply Prisma migration `20261011090000_chat_ui_drafts` before serving the new backend. It adds one `ChatUIDraft` per `(sessionId, viewId)` with source, typed JSON state, tool-call identity and revision. Deleting the chat cascades to its drafts.

Authenticated `GET` and `PUT /api/chat/sessions/{session_id}/ui-drafts/{view_id}` require chat ownership. A source must correspond to an actual published `ui_rendered` tool message. The most recent published version is writable even before the first draft save; historical cards cannot win an initial-save race. Writes lock the owned session and compare the expected revision. Concurrent edits receive 409 instead of overwriting another device. States reject nonfinite values, unknown request fields and payloads over 240 KB.

The browser scopes its cache to owner/session/view, serializes saves, retains unsynced edits, and reports sync or conflict errors with download/retry/reload actions. A clean older cache never replaces a newer server draft. Explicit reload replaces local edits. A three-way merge keeps edits where the source default is unchanged; changed model defaults win visibly. Stable row identities carry additions, deletions, reordering and selections. Model-provided IDs are essential; the system cannot infer that a newly named field is the same fact.

Regenerate the canonical catalog and parser bundle with `pnpm generate:openui`. Regenerate API hooks with `pnpm generate:api` after updating the OpenAPI artifact. File previews reuse existing storage and virus-scanning services; they do not add a media-generation provider. Execution details reuse existing authorization and graph APIs.

## Verification and limits

Integration tests run the actual chat renderer with API boundaries mocked. They cover linked edits/filters/totals, unknown inputs, hidden invalid rows, row selection, image retries and share privacy, repeat/conditional inputs, timezone ambiguity, itinerary exports, draft conflicts and execution stop/retry behavior. Backend tests use real PostgreSQL for draft ownership, concurrent writes, latest-version enforcement and cascade deletion. Canonical tests cover schema and calculation contracts separately from model output quality.

`backend/copilot/eval/openui/capability-journeys.jsonl` contains 36 scenarios and 44 turns, including negative cases, split into 24 development and 12 reserved checks. Criteria were frozen before generation. The authoring/implementing agent also reviews the outputs; this is exploratory evidence, not an independent benchmark. Desktop exports preserve raw accepted responses, failed attempts, runtime fingerprints and browser evidence. UI interaction tests are reported separately from model semantic judgments.

This catalog does not implement a spreadsheet formula language, arbitrary rich-text collaboration, new OAuth/provider flows, remote calendar writes, image editing/generation, map routing/geocoding, external task mutation or invented execution status. Those tasks use existing tools and approvals; missing data requires clarification. Model-written facts and source citations still need checking. Browser viewport checks do not establish native iOS/Android behavior.
