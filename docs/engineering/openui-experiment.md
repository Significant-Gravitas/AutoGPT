# OpenUI in Copilot

An opt-in integration of the [OpenUI React runtime](https://www.openui.com/docs/api-reference/react-lang) into AutoGPT's existing Copilot conversation. The model composes a bounded catalog of AutoGPT components: metrics, charts, searchable and sortable tables, checklists, editable forms, and follow-up buttons.

## Enable native generation

Set `CHAT_OPENUI_ENABLED=true` in the backend's gitignored `.env`, then restart the backend services. Open the normal `/copilot` conversation. Use your existing Copilot model, authentication, and billing configuration; no additional provider key or frontend generation endpoint is needed.

Example requests:

- “Use my agent run results to show an interactive performance report. Identify failures and offer a next step.”
- “Compare the leads from this CSV in a searchable table, with a chart of their scores.”
- “Help plan a product launch. Give me an editable brief before you build the plan.”

Otto first retrieves real data through its existing tools. When an interactive view helps, it discovers `tool:render_ui` and calls it through `run_capability`. The component schema is deferred, so ordinary turns do not carry the full library. Both the baseline and Claude SDK engines use their existing dispatch, permissions, stream, persistence, and cost accounting paths. The feature is off by default. Disabling it prevents new views; previously saved views remain readable.

## Conversation behavior

- Results render as full interactive cards in the message flow, outside collapsed tool chains. The renderer loads only when needed.
- Source, version, and a plain-text summary are saved as the normal tool result. Reopening a session restores the view.
- Edited fields and checklist selections stay in session storage for the same browser tab and exact source. They are not synced to another browser or written into the conversation until submitted.
- Form submission and follow-up buttons send an ordinary user message into the same Copilot conversation. This retains the normal queue, retry, billing, and approval behavior. A button never directly executes an agent or external action.
- Both frontend streaming implementations are supported, including AI SDK dynamic tool parts and persisted static tool parts.
- Pending views disable actions. Duplicate submissions are suppressed while sending; failed sends preserve edited inputs and allow retry.
- Shared conversations display the saved source without local drafts and disable conversation actions.
- Explore/Summary switches preserve inputs. Invalid or unsupported results show their saved summary and offer a normal conversational rebuild request. Rendering errors stay contained within the card.

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

- `frontend/src/lib/openui/catalog.ts` is the canonical Zod catalog. After editing it, run `node scripts/generate-openui.mts` in the frontend to update `backend/backend/copilot/tools/openui_library.txt`. A contract test checks exact parity; `--check` can verify it without writing.
- `backend/copilot/tools/render_ui.py` requires authentication and matching session ownership, checks the feature flag, bounds inputs, and returns a versioned result. It caps encoded output below the existing truncation thresholds. Full OpenUI syntax validation happens in the browser; the backend does not execute the program.
- `frontend/src/components/organisms/OpenUI` maps the real OpenUI renderer to AutoGPT primitives and Recharts. No generated JavaScript, HTML, arbitrary components, or tool execution is enabled. Completed programs containing queries or mutations are rejected.
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
