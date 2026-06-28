# Agent Mail Drafts
<!-- MANUAL: file_description -->
Blocks for creating, reviewing, editing, sending, and deleting email drafts in AgentMail. Drafts enable human-in-the-loop review, scheduled sending, and multi-step email composition workflows.
<!-- END MANUAL -->

## Agent Mail Create Draft

### What it is
Create a draft email for review or scheduled sending. Use send_at for automatic future delivery.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.inboxes.drafts.create(inbox_id, **params)` to create a new draft in the specified inbox. You must provide at least the recipient list; subject, text body, HTML body, cc, bcc, and in_reply_to are all optional.

If you supply `send_at`, the draft is scheduled for automatic delivery at that time and the returned `send_status` will be `scheduled`. If `send_at` is omitted, the draft remains unsent until you explicitly send it with the Send Draft block. Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to create the draft in | str | Yes |
| to | Recipient email addresses (e.g. ['user@example.com']) | List[str] | Yes |
| subject | Email subject line | str | No |
| text | Plain text body of the draft | str | No |
| html | Rich HTML body of the draft | str | No |
| cc | CC recipient email addresses | List[str] | No |
| bcc | BCC recipient email addresses | List[str] | No |
| in_reply_to | Message ID this draft replies to, for threading follow-up drafts | str | No |
| send_at | Schedule automatic sending at this ISO 8601 datetime (e.g. '2025-01-15T09:00:00Z'). Leave empty for manual send. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| draft_id | Unique identifier of the created draft | str |
| send_status | 'scheduled' if send_at was set, empty otherwise. Values: scheduled, sending, failed. | str |
| result | Complete draft object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Human-in-the-Loop Review** — Create a draft so a human can review and approve the email before it is sent.

**Scheduled Outreach** — Set `send_at` to queue a follow-up email that delivers automatically at a future date and time.

**Multi-Step Composition** — Create a draft with initial content, then use the Update Draft block to refine recipients or body text in later workflow steps.
<!-- END MANUAL -->

---

## Agent Mail Delete Draft

### What it is
Delete a draft or cancel a scheduled email. Removes the draft permanently.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.inboxes.drafts.delete(inbox_id, draft_id)` to permanently remove the specified draft. If the draft was scheduled for future delivery, deleting it also cancels that scheduled send.

On success the block yields `success=True`. Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to delete (also cancels scheduled sends) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | True if the draft was successfully deleted/cancelled | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Cancel Scheduled Send** — Delete a draft that was scheduled with `send_at` to prevent it from being delivered.

**Clean Up Rejected Drafts** — Remove drafts that a human reviewer has declined during an approval workflow.

**Abort Workflow** — Delete an in-progress draft when upstream conditions change and the email is no longer needed.
<!-- END MANUAL -->

---

## Agent Mail Get Draft

### What it is
Retrieve a draft email to review its contents, recipients, and scheduled send status.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.inboxes.drafts.get(inbox_id, draft_id)` to fetch a single draft by its ID. It returns the draft's subject, send status, scheduled send time, and the complete draft object.

Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| draft_id | Unique identifier of the draft | str |
| subject | Draft subject line | str |
| send_status | Scheduled send status: 'scheduled', 'sending', 'failed', or empty | str |
| send_at | Scheduled send time (ISO 8601) if set | str |
| result | Complete draft object with all fields | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Approval Gate** — Fetch a draft so a human reviewer can inspect its content and recipients before approving it for send.

**Schedule Monitoring** — Retrieve a scheduled draft to check its `send_status` and confirm it is still queued for delivery.

**Content Verification** — Read back a draft after creation or update to verify that the subject and body match expectations before proceeding.
<!-- END MANUAL -->

---

## Agent Mail List Drafts

### What it is
List drafts in an AgentMail inbox. Filter by labels=['scheduled'] to find pending sends.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.inboxes.drafts.list(inbox_id, **params)` to retrieve drafts from a single inbox. You can control page size with `limit`, paginate with `page_token`, and filter by `labels` (for example, `['scheduled']` to find only pending sends).

The block returns the list of draft objects, a count of drafts in the current page, and a `next_page_token` for fetching subsequent pages. Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to list drafts from | str | Yes |
| limit | Maximum number of drafts to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Filter drafts by labels (e.g. ['scheduled'] for pending sends) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| drafts | List of draft objects with subject, recipients, send_status, etc. | List[Dict[str, Any]] |
| count | Number of drafts returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inbox Dashboard** — List all drafts in an inbox to display a queue of pending emails awaiting review or approval.

**Scheduled Send Audit** — Filter by `labels=['scheduled']` to surface every draft that is queued for future delivery and verify send times.

**Batch Processing** — Paginate through all drafts in an inbox to perform bulk updates or deletions in a cleanup workflow.
<!-- END MANUAL -->

---

## Agent Mail List Org Drafts

### What it is
List all drafts across every inbox in your organization. Use for central approval dashboards.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.drafts.list(**params)` at the organization level, so it returns drafts across every inbox without requiring an inbox ID. You can control page size with `limit` and paginate with `page_token`.

The block returns the list of draft objects, a count of drafts in the current page, and a `next_page_token` for fetching subsequent pages. Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of drafts to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| drafts | List of draft objects from all inboxes in the organization | List[Dict[str, Any]] |
| count | Number of drafts returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Central Approval Dashboard** — List every pending draft across all inboxes so a manager can review and approve outbound emails in one place.

**Organization-Wide Analytics** — Count and categorize drafts across inboxes to report on email pipeline volume and bottlenecks.

**Stale Draft Cleanup** — Paginate through all org drafts to identify and delete old or abandoned drafts that were never sent.
<!-- END MANUAL -->

---

## Agent Mail Send Draft

### What it is
Send a draft immediately, converting it into a delivered message. The draft is deleted after sending.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.inboxes.drafts.send(inbox_id, draft_id)` to deliver the draft immediately. Once the send completes, the draft is deleted from the inbox and a message object is returned in its place.

The block yields the `message_id` and `thread_id` of the newly sent email along with the complete message result. Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to send now | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | Message ID of the now-sent email (draft is deleted) | str |
| thread_id | Thread ID the sent message belongs to | str |
| result | Complete sent message object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Post-Approval Dispatch** — Send a draft immediately after a human reviewer approves it in a review workflow.

**On-Demand Notifications** — Compose a draft earlier in the workflow and send it only when a triggering event occurs.

**Retry After Edit** — After updating a draft that failed validation, send it again without recreating it from scratch.
<!-- END MANUAL -->

---

## Agent Mail Update Draft

### What it is
Update a draft's content, recipients, or scheduled send time. Use to reschedule or edit before sending.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `client.inboxes.drafts.update(inbox_id, draft_id, **params)` to modify an existing draft. Only the fields you provide are changed; omitted fields are left untouched. Internally, `None` is used to distinguish between "omit this field" and "clear this field to empty," so you can selectively update recipients, subject, body, or scheduled send time without affecting other fields.

The block returns the updated `draft_id`, `send_status`, and the complete draft result. Any errors propagate to the block framework's global error handler, which yields them on the error output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to update | str | Yes |
| to | Updated recipient email addresses (replaces existing list). Omit to keep current value. | List[str] | No |
| subject | Updated subject line. Omit to keep current value. | str | No |
| text | Updated plain text body. Omit to keep current value. | str | No |
| html | Updated HTML body. Omit to keep current value. | str | No |
| send_at | Reschedule: new ISO 8601 send time (e.g. '2025-01-20T14:00:00Z'). Omit to keep current value. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| draft_id | The updated draft ID | str |
| send_status | Updated send status | str |
| result | Complete updated draft object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Reschedule Delivery** — Change the `send_at` time on a scheduled draft to delay or advance its delivery window.

**Reviewer Edits** — Allow a human reviewer to modify the subject or body of a draft before it is approved and sent.

**Dynamic Recipient Updates** — Update the recipient list based on data gathered in earlier workflow steps without recreating the draft.
<!-- END MANUAL -->

---
