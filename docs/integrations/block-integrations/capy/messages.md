# Capy Messages
<!-- MANUAL: file_description -->
Read what a Capy agent has said and steer it: send a follow-up instruction or an answer, or stop it mid-run.
<!-- END MANUAL -->

## Capy Interrupt Thread

### What it is
Stops the agent in a Capy thread mid-run, for example when it is heading the wrong way. The thread stays open for a new message.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/threads/{id}/interrupt`, which stops the agent's current work without adding a message. The thread stays open.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| interrupted | True once Capy accepted the stop | bool |

### Possible use case
<!-- MANUAL: use_case -->
Stop an agent that is spending credits on the wrong approach before sending it a corrected brief.
<!-- END MANUAL -->

---

## Capy List Thread Messages

### What it is
Reads a Capy thread's transcript: your brief, the agent's replies (including pull request links and questions), and optionally its tool steps. Returns the newest entries by default.

### How it works
<!-- MANUAL: how_it_works -->
Reads `GET /api/v1/threads/{id}/messages`. With no cursor it returns the newest entries; with `after_cursor` it returns only what came after it, so a loop that feeds `next_cursor` back in reads each new entry once. Tool entries are one-line activity summaries and are left out unless `include_tool_steps` is on.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |
| limit | Maximum number of transcript entries to return | int | No |
| after_cursor | Return only entries after this cursor (a previous call's next_cursor), oldest first. Leave empty for the newest entries. | str | No |
| include_tool_steps | Include the one-line tool activity entries alongside user and assistant messages | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| messages | Transcript entries, oldest first, each with id, source (user, assistant or tool), text, and created_at | List[Dict[str, Any]] |
| last_reply | The agent's most recent reply on this page, if any | str |
| next_cursor | Pass back as after_cursor to read only newer entries next time | str |

### Possible use case
<!-- MANUAL: use_case -->
Read the agent's full answer, or poll a long-running thread for new replies and forward them to a chat channel.
<!-- END MANUAL -->

---

## Capy Send Message

### What it is
Sends a message to the agent in a Capy thread: a follow-up instruction, a correction, or the answer to its question. The agent resumes work on it.

### How it works
<!-- MANUAL: how_it_works -->
Calls `POST /api/v1/threads/{id}/message`. `interrupt` (the default) stops current work and handles the message now, `steer` folds it into the work in progress, and `queue` waits for the current work to finish. Setting `model_id` switches the thread's model from this message on.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |
| text | The message for the agent | str | Yes |
| delivery | interrupt stops the current work and handles this message now; steer folds it into the work in progress; queue waits until the current work finishes | "interrupt" \| "steer" \| "queue" | No |
| model_id | Switch the thread to this Capy model ID. Empty keeps it. | str | No |
| reasoning | Reasoning effort for model_id. Needs model_id. | "" \| "none" \| "instant" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | ID of the admitted message; a queued one can be cancelled in Capy | str |
| deduped | True when Capy recognised this as a repeat of a message it already had | bool |

### Possible use case
<!-- MANUAL: use_case -->
Answer the question a Capy agent asked, or ask for a change after reviewing its pull request.
<!-- END MANUAL -->

---
