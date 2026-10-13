# OpenUI in Copilot

Copilot can answer with an interactive view inside the chat instead of plain text. The model writes a short program in [OpenUI Lang](https://github.com/thesysdev/openui) and calls the `render_ui` tool. The backend checks the program, and the frontend draws it with our own components. Generated code never runs: the program can only use the components listed below, and buttons and forms only send a message back to the same chat.

## Components

Every view starts with `root = Workspace(title, description, children, viewId)`.

| Group | Components |
| --- | --- |
| Layout | `Workspace`, `Form`, `FollowUp`, `Insight`, `Prose` |
| Display | `Metrics`/`Metric`, `SeriesChart`, `DataTable`, `Checklist`, `Timeline`, `Map`, `Artifact` |
| Interactive (inside `Form`) | `Itinerary`, `Comparison`, `ResultList`, `Field`, `TextAreaField`, `SelectField`, `MultiSelectField`, `NumberField`, `DateField`, `ToggleField` |

`Prose` holds up to 8,000 characters of Markdown. It renders with the same Markdown renderer as assistant messages, with raw HTML dropped.

The schemas live in `autogpt_platform/frontend/src/lib/openui/catalog*.ts`. The renderers live in `autogpt_platform/frontend/src/components/organisms/OpenUI/`.

## How validation works

1. `render_ui` (`backend/copilot/tools/render_ui.py`) rejects sources that do not start with `root = Workspace(`, are too long, or have unbalanced brackets.
2. It then runs `openui-validator.cjs`, a Node bundle of the frontend parser, on the source. The bundle checks syntax, every component's props against the Zod schemas, references, and form field names.
3. If the source is invalid, the model gets a fixable error and can retry. If it is valid, the tool returns the source and a plain-text summary built from the view. That summary is what later turns and saved chats read.
4. The browser parses the source again before rendering it.

The model sees the component list through the `source` parameter description, which comes from `backend/copilot/tools/openui_library.txt`. Prompt guidance for when and how to use views is in `backend/copilot/openui_prompt.py`.

## How drafts are stored today

Every new view needs a stable `viewId`. User edits (form values, ticked checklist items, itinerary changes) are saved per `(session, viewId)` in the `ChatUIDraft` table through `GET`/`PUT /api/chat/sessions/{session_id}/ui-drafts/{view_id}`. Writes carry a revision number, so a stale tab gets a 409 instead of overwriting newer edits. The browser also keeps a local cache of unsynced edits.

When the model revises a view with the same `viewId`, edits to fields the model did not change are kept, and the model's changes win where it did change a default. Older versions of a view become read-only. Shared chats show the saved source without edits and with actions turned off.

## Regenerating

After changing the catalog, run this in `autogpt_platform/frontend`:

```bash
pnpm generate:openui
```

It rewrites `openui_library.txt` and `openui-validator.cjs` in `autogpt_platform/backend/backend/copilot/tools/`. Commit both. `pnpm generate:openui --check` fails if they are out of date.

## Evals

`autogpt_platform/backend/backend/copilot/eval/openui/` holds frozen test journeys. See its README.
