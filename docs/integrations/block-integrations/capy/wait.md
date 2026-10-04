# Capy Wait
<!-- MANUAL: file_description -->
Wait for a Capy thread's agent to finish and collect its reply.
<!-- END MANUAL -->

## Capy Wait For Thread

### What it is
Waits for a Capy thread to finish (the agent delivered, asked a question or failed) and returns its status, latest reply and a link to the thread. Returns early when the timeout runs out; call it again to keep waiting. After Capy Send Message, pass its message_id so the wait ends on the reply to that message.

### How it works
<!-- MANUAL: how_it_works -->
Polls `GET /api/v1/threads/{id}` until the agent stops working, asks a question, or the timeout runs out, then reads the newest transcript entries for its latest reply. Capy reports a new or just-messaged thread as idle for a moment before the agent picks the message up, so the block only counts the thread as finished once something follows the last user message. After a follow-up, pass the `message_id` Capy Send Message returned as `after_message_id`: the wait then ends only on a reply to that message, which matters when the message was queued and the previous turn's reply is still the newest entry, and `last_reply` stays empty until it arrives. `model_id` and `billed_via` show the model the agent last ran on and who paid for it. Chat runs cancel a block call after five minutes, so keep `timeout_seconds` at 240 or less there and call the block again to keep waiting.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| thread_id | The Capy thread ID (starts with jam_) | str | Yes |
| timeout_seconds | How long to wait before returning the current state. Call the block again to keep waiting. Keep it at 240 or less when running from chat, which cancels a block call after 5 minutes. | int | No |
| after_message_id | The message_id Capy Send Message returned. The wait then ends on the agent's reply to that message, never an earlier one. | str | No |
| poll_interval_seconds | Seconds between status checks | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | The thread's latest state | Thread |
| thread_url | The thread in the Capy app, where its work shows live | str |
| status | working, waiting, idle, failed or archived | str |
| finished | True when the agent stopped working (it delivered, asked a question or failed); false when the timeout ran out first | bool |
| needs_you | True when the agent is waiting on an answer from a person | bool |
| last_reply | The agent's most recent reply, which carries its result, its question, or the pull request link. With after_message_id, only a reply to that message; empty until it answers. | str |
| model_id | The model the agent last ran on, e.g. supergrok/grok-4.5 | str |
| billed_via | Who pays for that model: the Capy balance, or the linked provider (Codex, Copilot, SuperGrok, Azure) | str |
| pull_request_url | The newest GitHub pull request link in the agent's recent replies, ready for the GitHub pull request blocks. Only emitted when the agent has linked one. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Deliver the Result**: Start a thread, wait for it, and pass the pull request link in `last_reply` to a Slack or email block.

**Answer Questions**: Stop waiting when the agent asks something and route the question to a person.

**Chat-Friendly Polling**: Wait in rounds of 240 seconds from AutoPilot, which cancels a block call after five minutes.
<!-- END MANUAL -->

---
