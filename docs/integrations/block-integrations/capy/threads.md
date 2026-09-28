# Capy Threads
<!-- MANUAL: file_description -->
Start Capy coding-agent threads, check on them, wait for their result and archive them. A thread is one task for a Capy agent running on a cloud machine against a project's repositories.
<!-- END MANUAL -->

## Capy Archive Thread

### What it is
Archives a Capy thread, taking it off the project board. Archived threads can be restored in the Capy app.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/threads/{id}/archive`. Archiving is reversible from the Capy app.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | The archived thread | Thread |
| archived | Whether the thread is now archived | bool |

### Possible use case
<!-- MANUAL: use_case -->
Tidy the project board once a thread's pull request has merged.
<!-- END MANUAL -->

---

## Capy Create Thread

### What it is
Starts a Capy cloud coding agent on a task in one of your Capy projects, such as fixing a bug, writing a feature or opening a pull request. Returns immediately with the thread ID; use Capy Wait For Thread to wait for the result. The run bills your Capy organization.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/threads` with the project, the brief and optional model, reasoning effort and machine size. Every call carries a `requestId` (generated when left empty), which Capy uses to dedupe, so a retried request can never start a second run. The block returns as soon as the thread exists; the agent keeps working in Capy's cloud.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | The Capy project to run in (see Capy List Projects) | str | Yes |
| message | The task for the agent, written as you would brief an engineer: the goal, where to look, what done looks like, and whether to open a pull request | str | Yes |
| title | Thread title. Leave empty to let Capy name it. | str | No |
| model_id | Capy model ID, e.g. openai/gpt-6-astra. Leave empty for the project's default model. | str | No |
| reasoning | Reasoning effort for the chosen model. Needs model_id. | "" \| "none" \| "instant" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" | No |
| machine_size | Machine size for the agent's VM. Empty uses Capy's default. | "" \| "small" \| "medium" \| "large" \| "ultra" \| "hyper" \| "bigguy" | No |
| request_id | Idempotency key. Re-sending the same request_id returns the thread it already created instead of starting a second run. Leave empty to generate one. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | The created thread | Thread |
| thread_id | ID of the created thread | str |
| status | The thread's status right after start | str |

### Possible use case
<!-- MANUAL: use_case -->
Hand a bug report from a support workflow to a coding agent with the instruction to fix it and open a pull request, then wait for the result with Capy Wait For Thread.
<!-- END MANUAL -->

---

## Capy Get Thread

### What it is
Gets a Capy thread's current status, title and credit usage, and whether it is still working or needs an answer.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/threads/{id}`. `is_active` is true while the status is `working` or `waiting`; `needs_you` is true when the agent has asked a question.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | The thread | Thread |
| status | working, waiting, idle, failed or archived | str |
| is_active | True while the agent is still working on the thread | bool |
| needs_you | True when the agent is waiting on an answer from a person | bool |

### Possible use case
<!-- MANUAL: use_case -->
Check whether a thread started earlier has finished before posting its result.
<!-- END MANUAL -->

---

## Capy List Threads

### What it is
Lists the agent threads in a Capy project with their status, most recently active first.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/threads` for one project and returns a page of threads, most recently active first. Pass `next_cursor` back as `cursor` for the next page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | The Capy project to list | str | Yes |
| limit | Maximum number of threads to return | int | No |
| cursor | Paging cursor from a previous call's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| threads | The threads on this page | List[Thread] |
| thread | Each thread, one at a time | Thread |
| next_cursor | Pass back as cursor for the next page; empty on the last | str |

### Possible use case
<!-- MANUAL: use_case -->
Build a daily digest of what Capy agents worked on in a project.
<!-- END MANUAL -->

---

## Capy Wait For Thread

### What it is
Waits for a Capy thread to finish (the agent delivered, asked a question or failed) and returns its status and latest reply. Returns early when the timeout runs out; call it again to keep waiting.

### How it works
<!-- MANUAL: how_it_works -->
Polls `GET /api/v1/threads/{id}` until the agent stops working, asks a question, or the timeout runs out, then reads the newest transcript entries for its latest reply. Capy reports a new or just-messaged thread as idle for a moment before the agent picks the message up, so the block only counts the thread as finished once something follows the last user message. Chat runs cancel a block call after five minutes, so keep `timeout_seconds` at 240 or less there and call the block again to keep waiting.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |
| timeout_seconds | How long to wait before returning the current state. Call the block again to keep waiting. Keep it at 240 or less when running from chat, which cancels a block call after 5 minutes. | int | No |
| poll_interval_seconds | Seconds between status checks | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | The thread's latest state | Thread |
| status | working, waiting, idle, failed or archived | str |
| finished | True when the agent stopped working (it delivered, asked a question or failed); false when the timeout ran out first | bool |
| needs_you | True when the agent is waiting on an answer from a person | bool |
| last_reply | The agent's most recent reply, which carries its result, its question, or the pull request link | str |

### Possible use case
<!-- MANUAL: use_case -->
Start a thread, wait for it, and pass the pull request link in `last_reply` to a Slack or email block.
<!-- END MANUAL -->

---
