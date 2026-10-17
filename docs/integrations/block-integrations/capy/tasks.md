# Capy Tasks
<!-- MANUAL: file_description -->
See how a Capy thread split its work across subagent tasks.
<!-- END MANUAL -->

## Capy List Thread Tasks

### What it is
Lists the subagent tasks a Capy thread fanned its work out to, with each task's status and credit spend. Read-only: steer a task by messaging its thread.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/threads/{id}/tasks`. Tasks come depth-first, so parents precede children; `task_path` such as `1.2` is the task's address from the thread root, and each task's `usage` covers its own subtree. Tasks are read-only: steer them by messaging the thread.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |
| limit | Maximum number of tasks to return | int | No |
| cursor | Paging cursor from a previous call's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tasks | Tasks depth-first; task_path is the dotted address from the thread root and usage is each task's own subtree spend | List[Task] |
| next_cursor | Pass back as cursor for the next page; empty on the last | str |

### Possible use case
<!-- MANUAL: use_case -->
**Spend Breakdown**: Report where a large thread spent its credits, task by task.

**Progress View**: Show which parts of a fanned-out job are done and which are still working.

**Failure Triage**: Find the subtask that failed before messaging the thread about it.
<!-- END MANUAL -->

---
