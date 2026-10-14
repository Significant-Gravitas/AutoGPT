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

## How edits are stored

Every new view needs a stable `viewId`. User edits (form values, ticked checklist items, itinerary changes) live in the `ChatUIDraft` table. Each row belongs to exactly one chat or one file; a check constraint enforces it.

- **Chat views** keep edits per `(session, viewId)` through `GET`/`PUT /api/chat/sessions/{session_id}/ui-drafts/{view_id}`. Writes carry a revision number, so a stale tab gets a 409 instead of overwriting newer edits. The browser also keeps a local cache of unsynced edits.
- **Saved views** keep edits on the file through `GET`/`PUT /api/workspace/files/{file_id}/view-state`. A write carries the edits revision and the file checksum it was made against; if either has moved on, it gets a 409, so edits never land on a newer program.

When the model revises a view with the same `viewId`, edits to fields the model did not change are kept, and the model's changes win where it did change a default. Older versions of a view become read-only. Shared chats show the saved source without edits and with actions turned off.

## Saved views

A view the user wants to keep is a workspace file: `<name>.openui`, MIME `application/vnd.autogpt.openui`, body the OpenUI program, metadata `{"openui": {"viewId", "originSessionId"}}`. It lives in the chat's folder (`/sessions/<id>/`), so it shows on the Files page and can be mentioned in later chats.

- `render_ui` saves one when given `file`: a new filename creates it, `workspace://<id>` revises that file. A view already saved from the chat is revised whenever it is rendered again with the same `viewId`. Revisions keep the file's ID and path, and `originSessionId` names the chat that last wrote it.
- `write_workspace_file` with a `.openui` name validates the program the same way, which is how experts keep notes without showing them in chat.
- `read_workspace_file` on a view returns its summary, its program and the user's current edits.
- `POST /api/chat/sessions/{session_id}/ui-drafts/{view_id}/save` saves a chat view from the "Save to files" button and moves its edits onto the file. Saving it again returns the same file.

The code is in `backend/copilot/tools/openui_files.py`, `backend/copilot/ui_view_files.py` and `backend/data/openui_files.py`.

## Regenerating

After changing the catalog, run this in `autogpt_platform/frontend`:

```bash
pnpm generate:openui
```

It rewrites `openui_library.txt` and `openui-validator.cjs` in `autogpt_platform/backend/backend/copilot/tools/`. Commit both. `pnpm generate:openui --check` fails if they are out of date.

## Evals

`autogpt_platform/backend/backend/copilot/eval/openui/` holds frozen test journeys. See its README.
