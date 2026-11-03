# TASKS_V1 contract (backend ⇄ frontend)

Owner: BACKEND agent. Frontend builds against this. Changes are appended under "Changelog" at the bottom.

- Feature flag: backend `Flag.TASKS_V1 = "tasks-v1"`, also added to `FrontendFlag` (so after `pnpm generate:api` the frontend has `Flag.TASKS_V1` for `useGetFlag`; add `[Flag.TASKS_V1]: false` to `defaultFlags`). Locally forced on with `FORCE_FLAG_TASKS_V1=true` (backend env) / `NEXT_PUBLIC_FORCE_FLAG_TASKS_V1=true` (frontend env).
  When off, every `/api/tasks/*` route returns **404**. Frontend: call `GET /api/tasks/enabled` → `{ "enabled": bool }` (never 404s) to decide whether to show the Tasks nav / Home "Needs you" / plan card. (Or use `useGetFlag(Flag.TASKS_V1)`; treat either as on.)
- Router: FastAPI, `prefix="/api/tasks"`, `tags=["v2","tasks"]`. Every route has an explicit `operation_id` (listed below) so `pnpm generate:api` yields hooks like `useGetV2ListTasks`. Until the backend is up, hand-written fetches against these paths are fine.
- Auth: standard user JWT (same as other `/api` routes). All data is per-user.
- Copy: product copy never says "agent"/"graph". Assignee kinds are shown as "Me", the expert's name, or "Otto".

## Enums

```
TaskStatus   = "backlog" | "todo" | "in_progress" | "blocked" | "in_review" | "done" | "cancelled"
               // "blocked" is rendered as "Needs you". Board columns: todo | in_progress | blocked | done.
               // backlog & cancelled only when the "Hidden" toggle is on.
TaskPriority = "none" | "low" | "medium" | "high" | "urgent"
AssigneeType = "user" | "expert" | "otto"
ActorType    = "user" | "expert" | "otto" | "system"
```

## Objects

### Assignee / Actor
```jsonc
{ "type": "expert", "id": "uuid|null", "name": "Maria", "avatar_url": "https://…|null", "color": "sky-300|null" }
// type=user → id=user id, name="Me"; type=otto → id=null, name="Otto"
```

### Task (list item)
```jsonc
{
  "id": "uuid",
  "number": 123,
  "key": "T-123",
  "title": "Draft launch email",
  "description": "markdown|null",
  "status": "in_progress",
  "priority": "high",
  "assignee": Assignee | null,
  "due_date": "2026-11-05|null",            // ISO date
  "labels": ["launch"],
  "parent_id": "uuid|null",
  "parent_key": "T-120|null",
  "outcome_summary": "markdown|null",       // set when done
  "work_session_id": "chat session uuid|null", // the expert's dedicated work thread (link: /copilot?sessionId=…)
  "budget_cap_usd": 2.0,                    // null = default cap (2.00)
  "review_required": false,
  "metrics": { "turns": 3, "tool_calls": 11, "tokens": 48211, "cost_usd": 0.12, "duration_ms": 201000 },
  "subtask_counts": { "total": 3, "done": 1 },
  "pending_input": PendingInput | null,     // present while status=blocked waiting for the user
  "live": { "running": true, "current_step": "Searching the web", "turn_started_at": "ISO|null" },
  "created_at": "ISO", "updated_at": "ISO", "started_at": "ISO|null", "completed_at": "ISO|null"
}
```

### PendingInput
```jsonc
{ "id": "uuid", "question": "Which tone?", "options": ["Formal", "Casual"], "asked_by": Actor, "asked_at": "ISO" }
// options may be []  → free-text answer
```

### TaskDetail = Task + 
```jsonc
{
  "subtasks": Task[],
  "attachments": Attachment[],
  "parent": { "id": "uuid", "key": "T-120", "title": "…" } | null
}
```

### Attachment (linked artifact)
```jsonc
{ "id": "uuid", "kind": "file" | "chat" | "run" | "url" | "text",
  "title": "launch-email.md", "ref_id": "workspace file id / session id / graph exec id | null",
  "url": "https://… | null", "created_at": "ISO", "added_by": Actor }
```

### ActivityEvent
```jsonc
{
  "id": "uuid", "task_id": "uuid", "created_at": "ISO", "actor": Actor,
  "kind": "created" | "status_changed" | "assigned" | "comment" | "turn" | "input_requested"
        | "input_answered" | "attachment_added" | "subtask_created" | "outcome" | "plan_confirmed" | "system",
  "body": "markdown|null",          // comment text, question, answer, outcome summary, system note
  "data": { /* kind-specific, see below */ }
}
```
kind-specific `data`:
- `status_changed`: `{ "from": TaskStatus, "to": TaskStatus }`
- `assigned`: `{ "from": Assignee|null, "to": Assignee|null }`
- `comment`: `{ "mentions": [{ "expert_id": "uuid", "name": "Maria" }] }`
- `turn` (one collapsed row per expert turn — render `"4 tool calls · 12k tokens · $0.04 · 1m 20s"`):
  ```jsonc
  { "session_id": "uuid", "turn_index": 2, "status": "running" | "completed" | "failed",
    "tool_calls": 4, "tokens": 12034, "input_tokens": 10000, "output_tokens": 2034,
    "cost_usd": 0.04, "duration_ms": 80000,
    "summary": "Drafted the email|null",
    "tool_call_cards": [ { "id": "toolu_…", "name": "web_search", "title": "Web search",
                           "status": "completed" | "error" | "running",
                           "input_preview": "string|null", "output_preview": "string|null" } ] }
  ```
- `input_requested`: `{ "input_id": "uuid", "options": [] }` (body = question)
- `input_answered`: `{ "input_id": "uuid" }` (body = answer)
- `attachment_added`: `{ "attachment": Attachment }`
- `subtask_created`: `{ "subtask_id": "uuid", "subtask_key": "T-124", "title": "…" }`
- `outcome`: `{}` (body = outcome summary)

### Expert option (assignee picker / @mentions)
```jsonc
{ "id": "uuid", "name": "Maria", "role": "Marketing", "avatar_url": "…|null", "color": "…|null",
  "busy_task_id": "uuid|null", "busy_task_key": "T-12|null" }   // busy = already working a task (1 at a time)
```

### Plan (Otto plan card)
```jsonc
{
  "id": "uuid", "status": "proposed" | "confirmed" | "cancelled",
  "goal": "Launch the newsletter", "budget_usd": 5.0,
  "steps": [ { "title": "Draft copy", "description": "…|null", "assignee": Assignee } ],
  "session_id": "chat session uuid|null",
  "parent_task_id": "uuid|null",           // set after confirm
  "task_ids": ["uuid"],                     // sub-task ids after confirm
  "created_at": "ISO"
}
```

## REST routes (`/api/tasks`)

| Method | Path | operation_id | Body / query | Returns |
|---|---|---|---|---|
| GET | `/api/tasks/enabled` | `get_tasks_enabled` | – | `{enabled}` |
| GET | `/api/tasks` | `list_tasks` | query: `status` (comma list), `assignee_type`, `assignee_id`, `priority` (comma list), `due` (`overdue`\|`today`\|`week`\|`none`), `parent_id`, `top_level` (bool, default false), `mine` (bool: assigned to me), `include_hidden` (bool, default false → excludes backlog/cancelled unless `status` given), `q` (title search), `limit` (default 200) | `{ "tasks": Task[] }` |
| POST | `/api/tasks` | `create_task` | `{ title, description?, status?, priority?, assignee_type?, assignee_id?, due_date?, labels?, parent_id?, budget_cap_usd?, review_required? }` | `TaskDetail` (201). Assigning to an expert/Otto auto-starts work (status → in_progress). Default status: `todo`. |
| GET | `/api/tasks/needs-you` | `list_tasks_needing_me` | – | `{ "tasks": Task[] }` (status=blocked, plus in_review) |
| GET | `/api/tasks/experts` | `list_task_experts` | – | `{ "experts": ExpertOption[] }` |
| GET | `/api/tasks/{task_id}` | `get_task` | – | `TaskDetail` |
| PATCH | `/api/tasks/{task_id}` | `update_task` | any subset of create fields + `outcome_summary` | `TaskDetail`. Changing assignee to an expert auto-starts. |
| DELETE | `/api/tasks/{task_id}` | `delete_task` | – | 204 |
| POST | `/api/tasks/{task_id}/subtasks` | `create_subtask` | create body (parent forced; parent must itself be top-level → 400 otherwise) | `TaskDetail` |
| GET | `/api/tasks/{task_id}/activity` | `list_task_activity` | – | `{ "events": ActivityEvent[] }` oldest→newest |
| POST | `/api/tasks/{task_id}/comments` | `create_task_comment` | `{ body }` — `@Name` mentions of experts are parsed server-side; mentioning an expert on a task wakes that expert if it's the assignee | `ActivityEvent` |
| POST | `/api/tasks/{task_id}/answer` | `answer_task_input` | `{ answer, input_id? }` | `TaskDetail` (status → in_progress, expert resumes) |
| POST | `/api/tasks/{task_id}/start` | `start_task` | – | `TaskDetail` (manually (re)start expert work) |
| POST | `/api/tasks/{task_id}/attachments` | `add_task_attachment` | `{ kind, title, ref_id?, url? }` | `Attachment` |
| GET | `/api/tasks/plans/{plan_id}` | `get_task_plan` | – | `Plan` |
| POST | `/api/tasks/plans/{plan_id}/confirm` | `confirm_task_plan` | `{ steps? }` (optional edited steps) | `{ "plan": Plan, "parent_task": TaskDetail }` — creates parent task (Otto) + one sub-task per step, auto-starts the step tasks |
| POST | `/api/tasks/plans/{plan_id}/cancel` | `cancel_task_plan` | – | `Plan` |

Errors: 404 unknown task / flag off; 400 validation (e.g. sub-task of a sub-task); 409 not used (busy experts queue instead: task stays `todo` with a `system` activity "Waiting for Maria to finish T-12", starts automatically later).

## Chat (Otto) plan card

Otto's copilot tool `propose_plan` returns a tool output with `"type": "task_plan"` and `"plan": Plan`. In the chat UI render tool calls named **`propose_plan`** (output `type === "task_plan"`) as the plan card: steps, expert per step, budget, **Approve & start** → `POST /api/tasks/plans/{plan.id}/confirm`. After confirm, `confirm_task_plan` tool output (`"type": "task_plan_confirmed"`, `plan`, `parent_task`) can render as a compact "Plan started · T-130" card.

## Live updates (existing notification WebSocket)

Delivered over the existing user notification bus → WS message `{ "method": "notification", "data": Payload }`:

```jsonc
{ "type": "task_event",
  "event": "task_created" | "task_updated" | "task_deleted" | "task_activity" | "task_needs_input" | "task_completed",
  "task_id": "uuid", "task_key": "T-123", "title": "…",
  "task": Task | null,              // fresh snapshot (null on delete)
  "activity": ActivityEvent | null  // for task_activity / task_needs_input / task_completed
}
```
- `task_needs_input` → toast "T-123 needs you: <question>" + Home "Needs you" refresh.
- `task_completed` → toast "T-123 is done".
- Live strip: `task_updated` with `task.live` changes (current step / running) and `task_activity` with `kind:"turn"` (status running→completed) while the turn streams.
Frontend: on any `task_event`, invalidate the tasks list/detail/activity queries (or patch from `task`).

## Expert / Otto copilot tools (backend only, for reference)
`list_my_tasks`, `update_task_status`, `comment_on_task`, `request_input`, `link_artifact`, `create_task` (sub-task, 1 level), `propose_plan`, `confirm_task_plan`.

## Changelog
- v1 (initial).
- v1.1: create/update also accept `assignee: {type, id}` (alias of `assignee_type`/`assignee_id`) and `budget_usd` (alias of `budget_cap_usd`); `POST /api/tasks/plans/{plan_id}/approve` aliases `/confirm`.
