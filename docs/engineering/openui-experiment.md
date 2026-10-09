# OpenUI in Copilot

An opt-in integration of the [OpenUI React runtime](https://www.openui.com/docs/api-reference/react-lang) into AutoGPT's existing Copilot conversation. The model composes a bounded catalog of 17 AutoGPT components, including interactive maps, timelines, metrics, bar/line/donut charts, searchable and sortable tables, checklists, typed forms, and follow-up buttons.

## Enable native generation

Set `CHAT_OPENUI_ENABLED=true` in the backend's gitignored `.env`, then restart the backend services. Open the normal `/copilot` conversation. Use your existing Copilot model, authentication, and billing configuration; no additional provider key or frontend generation endpoint is needed.

Example requests:

- “Use my agent run results to show an interactive performance report. Identify failures and offer a next step.”
- “Compare the leads from this CSV in a searchable table, with a chart of their scores.”
- “Help plan a product launch. Give me an editable brief before you build the plan.”
- “Map these customer locations and propose a visit timeline. Let me change the travel mode, date, and budget in this chat.” Supply coordinates or let Copilot retrieve them through its existing tools.

Otto first retrieves real data through its existing tools. When an interactive view helps, it discovers `tool:render_ui` and calls it through `run_capability`. The component schema is deferred, so ordinary turns do not carry the full library. Both the baseline and Claude SDK engines use their existing dispatch, permissions, stream, persistence, and cost accounting paths. The feature is off by default. Disabling it prevents new views; previously saved views remain readable.

## Conversation behavior

- Results render inline in the message flow, outside collapsed tool chains. There is no outer card, presentation tab, or nested vertical scroller. The renderer loads only when needed.
- Source, version, and a plain-text summary are saved as the normal tool result. Reopening a session restores the view.
- Edited fields and checklist selections stay in session storage for the same browser tab and exact source. They are not synced to another browser or written into the conversation until submitted.
- Checklist items can include `done: true` for completion the user has already reported. Saved local edits override these initial values, including an explicitly cleared checklist. Only the checklist's own count reacts to local toggles; model-written summary metrics remain a snapshot of the conversation.
- Form submission and follow-up buttons send an ordinary user message into the same Copilot conversation. This retains the normal queue, retry, billing, and approval behavior. A button never directly executes an agent or external action.
- Both frontend streaming implementations are supported, including AI SDK dynamic tool parts and persisted static tool parts.
- Pending views disable actions. Duplicate submissions are suppressed while sending; failed sends preserve edited inputs and allow retry.
- Shared conversations display the saved source without local drafts and disable conversation actions.
- Invalid or unsupported results show their saved summary and offer a normal conversational rebuild request. Rendering errors stay contained within the response. Shared-view and send status appear only when relevant.

## Geographic and planning views

`Map` displays up to 50 supplied locations with numbered, keyboard-accessible markers, zoom controls, category/text filtering, a matching place list, and selected-place details. **Discuss this place** submits that location's name, coordinates, category, and detail through the existing conversation. Selection survives a reload in the same tab. Map exploration remains available in shared conversations; discussion actions are disabled there.

The Leaflet bundle loads only for a completed map. Basemap tiles come from OpenStreetMap with visible attribution and browser caching, following its [tile policy](https://operations.osmfoundation.org/policies/tiles/). The map does not request geolocation, geocode addresses, calculate routes, or prefetch offline tiles. Coordinates must come from the user or tools. Place names and details are rendered as text and are not sent to the tile service. If tiles fail, the searchable list and conversation actions remain usable. Review the tile provider's capacity and terms before a broader deployment.

`Timeline` shows dated or timed milestones with done/current/planned states. `TrendChart` supports ordered observations, including negative values; `DonutChart` shows category totals and percentages. Forms can combine `Field`, `SelectField`, `DateField`, and `NumberField`.

Numeric fields retain the typed draft, including incomplete or invalid text, and show a local accessible warning for nonnumeric values, nonfinite numbers, bounds, and increments. Required text and date fields also show local feedback. Invalid submissions focus the first invalid control and never send a chat message. Corrected numeric values are converted to numbers immediately before submission. Drafts remain in the same browser tab; validation errors are not sent to the model. These controls only collect preferences; they do not schedule or execute work.

## Sample lab

Use Node 24 and the repository's pnpm version in `autogpt_platform/frontend`:

```bash
pnpm install
pnpm generate:api
NEXT_PUBLIC_OPENUI_EXPERIMENT=true pnpm exec next dev --turbo --port 3000
```

Open **http://localhost:3000/tour/openui** for prepared examples without a backend or account. **http://localhost:3000/copilot/openui** uses the authenticated platform shell. The optional **UI Lab** sidebar entry uses this frontend flag, which is separate from the backend generation flag and takes effect at build time.

The lab demonstrates performance → failure investigation, lead research → outreach planning, and a campaign brief → personalized checklist. All sample data is fictional and labeled. It replays prepared programs, never calls a model or executes agents, and explains unsupported prompts. Use **Continue in Copilot** for actual generation. Interactive/Text response/Source compares the same sample; Export downloads its OpenUI program. The lab resets on refresh.

## Implementation

- `frontend/src/lib/openui/catalog.ts`, `catalog-sections.ts`, and `catalog-fields.ts` define the canonical Zod catalog. Run `pnpm generate:openui` in the frontend to update `backend/backend/copilot/tools/openui_library.txt`. A contract test checks exact parity; `pnpm generate:openui --check` verifies it without writing.
- `backend/copilot/tools/render_ui.py` requires authentication and matching session ownership, checks the feature flag, bounds inputs, and returns a versioned result. It caps encoded output below the existing truncation thresholds. A bounded lexical check rejects unfinished or mismatched delimiters before publishing a view and returns a repairable tool error. This check respects quoted strings and line comments; it is not a full parser. Component schemas, references, and full OpenUI syntax are still validated in the browser. The backend does not execute the program.
- `backend/copilot/openui_prompt.py` contains opt-in task-to-view guidance. It asks the model to verify dated plans, reconcile totals and dependencies, invalidate derived values after constraint changes, preserve reported progress, and use ordinary text or clarification when an interactive view adds no value. Guidance does not guarantee factual correctness or valid programs.
- `frontend/src/components/organisms/OpenUI` maps the real OpenUI renderer to AutoGPT primitives, Recharts, and Leaflet. No generated JavaScript, HTML, arbitrary components, or tool execution is enabled. Completed programs containing queries or mutations are rejected, and component props are validated against the canonical Zod schemas, including geographic bounds and typed field defaults.
- `frontend/src/app/(platform)/copilot/tools/RenderUI` handles native result rendering, drafts, summary fallback, and conversation follow-ups.
- Existing `/copilot` authentication and shared-session rules apply; no new protected prefix is introduced.
- OpenUI is pinned at `0.3.0`. Both Next.js bundlers alias upstream's automatic development widget to a local no-op, avoiding its CDN script and promotional injection while preserving the renderer.

## Verification

```bash
# frontend
pnpm format
pnpm lint
pnpm types
pnpm test:unit

# backend
poetry run pytest backend/copilot/tools/render_ui_test.py backend/copilot/tools/tool_schema_test.py
poetry run format
```

Tests exercise real Copilot host streaming and follow-up POSTs on both transport implementations, saved session conversion, shared views, draft restoration under React Strict Mode, failed and duplicate sends, malformed output, and catalog parity. Backend tests exercise discovery, opt-in prompting, authentication, ownership, encoded limits, and the real baseline/SDK dispatch envelopes. Automated tests use prepared model output; smoke-test each deployment against its configured provider.

On October 8, 2026, a private deployment using an authenticated ChatGPT/Codex connection generated a launch review from supplied fictional data in the normal chat. The response included metrics, a chart, a searchable comparison table, a recommendation, a checklist, and an editable form. Changing the audience and budget and submitting the form produced an updated plan in the same conversation. Both generated views, edited inputs, and checklist selections survived a page reload. Table filtering and the actual generated forms were also checked at a 390-pixel viewport. This test found and fixed default field initialization overwriting restored drafts when Strict Mode replays effects; defaults now check the current form state before writing.

The expanded catalog was also tested against the connected model that day. It generated a three-place Chicago map, proposed timeline, and typed preference form from supplied data. All basemap tiles loaded, marker selection and category filtering worked, and **Discuss this place** produced an updated comparison in the same session. Changing travel mode to transit, date to October 12, and budget to 240 survived a reload and produced an updated plan when submitted. A subsequent response rendered the map with a line chart including negative observations and a donut breakdown with the correct total and percentages. Desktop and 390-pixel layouts were inspected without horizontal overflow.

Additional automated coverage exercises real Leaflet markers, tile-failure fallback, map selection restoration, shared-view action restrictions, typed field restoration/submission under Strict Mode, coordinate/date/default validation, antimeridian bounds, and treating marker names as text. The changed backend files pass their focused type check and all 291 tool/schema tests. The repository-wide backend `poetry run format` reaches Pyright but reports errors in the unchanged `scripts/backfill_store_sub_headings.py` (`LiteralString`) and three webapp-testing fixture examples (missing Playwright imports).

Frontend format, lint, and type checks and all pre-commit hooks passed. The full frontend run covered 876 files and 9,546 tests: 9,540 passed, two expected failures, and four failures in unchanged chat/onboarding tests. All 46 tests in those four files passed on a sequential rerun without code changes. The configured backend pre-commit type check also passed; the broader formatter check above includes additional scripts and fixture examples.

### October 9: inline validation and 80-case evaluation

The native response now has no Explore/Summary tabs, Interactive view header, outer frame, or response-level vertical scrollbar. Numeric fields preserve invalid drafts and validate locally on blur or submission. Integration tests cover letters, empty values, finite numbers, bounds, increments, signed decimals, IME composition, date feedback, restored drafts, focus, and typed submission. In a live browser, `abc` produced a local warning without creating a user message; correcting it to `25` submitted a JSON number and received a real model response in the same session.

A fixed collection of 80 prompts was run once before and once after improving model guidance: ten each for maps, charts, tables, plans, forms, dashboards, plain-text controls, and capability boundaries. Both conditions used fresh authenticated Copilot sessions with the connected Codex route and standard tier. The session API did not expose the resolved upstream model identifier. Only one positive-control prompt named `render_ui`; all other requests used ordinary task language. The renderer and local validation were identical in both conditions. The treatment changed task-to-component guidance and exposed component size limits in the generated tool library.

Automated checks improved from **68/80 to 80/80**. The baseline missed eleven visual opportunities and emitted one oversized checklist. Revised outputs preserved all 15 checklist tasks in two valid groups; all ten plain-text controls remained text. All 17 catalog components appeared in the suite. The actual React renderer mounted 56 baseline views and verified one summary fallback; it mounted all 68 revised views, including successful submissions from 15 forms and local invalid-number rejection. Browser samples covered desktop and 390-pixel layouts, a corrected live submission, negative chart values, antimeridian map wrapping, and literal-text table filtering.

Manual review of every revised prompt, source, and reply recorded **67 clean, 12 partial, and one issue**, independently of the automated score. The issue is generated copy calling a text field optional when `Field` is currently required. Other findings include static checklist descriptions and metrics becoming stale as local state changes, categorical time axes, missing signed category bars, no draggable kanban or live routing, redundant columns, and misleading Save labels on chat submissions. A polar-coordinate probe should cover the distinction between geographic latitude and Web Mercator bounds. These are remaining experiment limitations, not capabilities certified by the automated pass rate.

The Desktop collection `OpenUI-evals-2026-10-09` contains the prompts, raw outputs, prompt/catalog fingerprints, per-case reviews, searchable HTML report, CSV/JSON exports, screenshots, and replay scripts. `frontend/scripts/evaluate-openui.ts` grades recorded runs against the canonical parser and schemas. The collection's separate fidelity scorer checks supplied values, coordinates, constraints, and record counts. One definition control contains conflicting supplied-information-only wording and is scored only for UI selection. A post-stream DNS collection failure was recovered by reading the existing session; two startup failures were deferred until the backend was ready. No completed model sample was replaced. This is a single-sample exploratory evaluation with synthetic data and review by the implementing agent, not a blinded or statistically significant benchmark.

Frontend format, lint, types, and 73 focused tests passed. The full run had 9,553 passes, two expected failures, a catalog contract failure during regeneration, and an unchanged compaction test timeout; both failures passed in the focused sequential run. Backend coverage included 328 prompting/tool/schema tests: the only initial failure measured the additional 426 deferred catalog characters; after recording that intentional increase, all 279 schema tests passed. Configured pre-commit checks passed, with the Prettier hook rerun after correcting the managed job's PATH. The broader backend formatter retains the unrelated errors documented above.

### October 9: realistic journeys and repeated improvement

The subsequent collection adds 24 user journeys with 39 turns: 12 positive, nine negative, and three boundary cases. It tests actual outcomes and changing constraints, including the SF museum trip, wheelchair access, apartment move-in costs, workshop timing, dietary catering, study availability, missing supplier quotes, and conflicting order quantities. The frozen prompts and criteria are in `backend/copilot/eval/openui/journeys.jsonl`. These complement the earlier component checks; the earlier automated 80/80 score did not establish that real plans were correct.

The baseline and selected candidate each ran twice across all 24 cases; an intermediate candidate ran twice across 16 development cases. All 128 journey runs and 207 submitted turns are retained, including one credential-busy failure and its unsubmitted dependent turn. The selected candidate was locked before reviewing response contents from the eight final-check cases. Those cases were authored by the same implementing agent and held apart from tuning, rather than being an independent or unseen benchmark.

Complete-journey pass requires every task criterion, appropriate UI selection, canonical parser/schema validity, and completion across all turns. Development passes were **19/32 → 19/32 → 21/32** across baseline and two iterations; final-check passes were **9/16 → 12/16**. Development UI choice improved **29/32 → 32/32**, while parser/completion stayed **29/32**. All six final-check negative runs avoided unnecessary UI in both versions. Two repeats and implementing-agent judgments provide exploratory evidence, not a production reliability claim; the case-cluster bootstrap intervals include no improvement.

The retained changes strengthen ordinary-task guidance, initialize real checklist boxes from user-reported completion, preserve local overrides, and reject incomplete source delimiters before publishing a view. Tests observed the checkbox and malformed-source failures before their fixes. The backend preflight is deliberately bounded: component limits, references, and full language validation still belong to the frontend parser.

Important failures remain. Both selected SF journeys fail at least one required outcome, including an early museum arrival or wrong streetcar direction. Apartment revisions keep a stale driving commute after car availability changes. Scheduling replies can contain correct primary timelines with incorrect buffer arithmetic. Seven-column tables and duplicate definitions still cause fallback rendering. The Paris final-check preserves the requested Tuesday but leaves replacement museum hours unverified after blocked lookups. One meal-plan revision took 469 seconds. Some strict rubric failures also reflect omitted clarification or alternative suggestions in otherwise safe concise answers; those judgments are explicit in the report.

Actual browser checks verified map filtering and **Discuss this place** through a real model reply in the same chat, persisted checkbox edits, local invalid-number feedback without a submission, reload, and desktop/390-pixel layouts. Static summary metrics do not react to checkbox edits, and existing floating chat controls can overlap content near the top on mobile.

The Desktop folder `OpenUI-hill-climb-2026-10-09` contains a searchable HTML report, the baseline and both candidates, raw visible outputs, frozen criteria, individual judgments, parser grades, model-route observations, source references, screenshots, reproducible scripts, and protocol disclosures. It preserves every failed accepted sample. Authentication values and hidden reasoning are excluded.

Frontend format, lint, types, and configured pre-commit hooks passed. The full frontend suite recorded 9,550 passes, nine failures and two expected failures across 878 files; all six affected files passed a sequential rerun (57/57) without code changes. All four new checklist integration tests and 339 focused backend tests passed, along with scoped backend Pyright. The broader backend formatter retains the four unrelated errors noted above.
