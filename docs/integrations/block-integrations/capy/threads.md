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
**Board Tidy-Up**: Archive a thread once its pull request has merged.

**Workflow Cleanup**: Archive the threads a scheduled job created after reporting their results.

**Test Hygiene**: Archive throwaway threads started to check a model or project.
<!-- END MANUAL -->

---

## Capy Get Thread

### What it is
Gets a Capy thread's current status, title and credit usage, whether it is still working or needs an answer, and a link to watch it.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/threads/{id}`. `is_active` is true while the status is `working` or `waiting`; `needs_you` is true when the agent has asked a question. `model_id` is the model the agent last ran on and `billed_via` says whether that bills the Capy balance or a linked provider.
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
| thread_url | The thread in the Capy app, where its work shows live | str |
| status | working, waiting, idle, failed or archived | str |
| is_active | True while the agent is still working on the thread | bool |
| needs_you | True when the agent is waiting on an answer from a person | bool |
| model_id | The model the agent last ran on, e.g. supergrok/grok-4.5 | str |
| billed_via | Who pays for that model: the Capy balance, or the linked provider (Codex, Copilot, SuperGrok, Azure) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Status Check**: See whether a thread started earlier is still working before posting its result.

**Question Detection**: Route a thread to a person when `needs_you` turns true.

**Cost Attribution**: Report which model a thread ran on and whether it billed the Capy balance or a linked subscription.
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
**Daily Digest**: Summarise what Capy agents worked on in a project today.

**Find Earlier Work**: Locate the thread for "the Node upgrade Capy did last week" to send it a follow-up.

**Stale Thread Sweep**: Find idle threads to archive once their pull requests have merged.
<!-- END MANUAL -->

---
