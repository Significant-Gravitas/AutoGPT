# AgentMail Drafts
<!-- MANUAL: file_description -->
Blocks for creating, reviewing, editing, sending, and deleting email drafts in AgentMail. Drafts enable human-in-the-loop review, scheduled sending, and multi-step email composition workflows.
<!-- END MANUAL -->

## Create Draft

### What it is
A block that creates a draft email in an AgentMail inbox for review or scheduled sending.

### How it works
<!-- MANUAL: how_it_works -->
The block creates an unsent email that can be reviewed, edited, and sent later. Use `send_at` to schedule automatic sending at a future time (ISO 8601 format). Scheduled drafts are auto-labeled 'scheduled' and can be cancelled by deleting the draft. Supports CC/BCC recipients and reply threading via `in_reply_to`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to create the draft in | str | Yes |
| to | Recipient email addresses | List[str] | Yes |
| subject | Email subject line | str | No |
| text | Plain text body of the draft | str | No |
| html | Rich HTML body of the draft | str | No |
| cc | CC recipient email addresses | List[str] | No |
| bcc | BCC recipient email addresses | List[str] | No |
| in_reply_to | Message ID this draft replies to, for threading | str | No |
| send_at | Schedule automatic sending at this ISO 8601 datetime (e.g. '2025-01-15T09:00:00Z') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| draft_id | Unique identifier of the created draft | str |
| send_status | 'scheduled' if send_at was set, empty otherwise | str |
| result | Complete draft object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Human-in-the-Loop Review**: AI agent drafts an email, human reviews and approves it before sending.

**Scheduled Outreach**: Create drafts scheduled to send at optimal times for recipient engagement.
<!-- END MANUAL -->

---

## Get Draft

### What it is
A block that retrieves a specific draft from an AgentMail inbox for review.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches the draft contents including recipients, subject, body, and scheduled send status. Use this to review a draft before approving or editing it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| draft_id | Unique identifier of the draft | str |
| subject | Draft subject line | str |
| send_status | Scheduled send status: 'scheduled', 'sending', 'failed', or empty | str |
| send_at | Scheduled send time (ISO 8601) if set | str |
| result | Complete draft object with all fields | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Draft Review**: Retrieve a draft to display its contents in an approval dashboard before sending.

**Schedule Verification**: Check the scheduled send time of a draft to confirm it will be delivered at the right time.
<!-- END MANUAL -->

---

## List Drafts

### What it is
A block that lists all drafts in an AgentMail inbox with optional label filtering.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves a paginated list of drafts from the specified inbox. Use labels=['scheduled'] to find all drafts queued for future sending. Supports pagination for inboxes with many drafts.
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
| drafts | List of draft objects with subject, recipients, send_status, etc. | List[dict] |
| count | Number of drafts returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Approval Dashboard**: List all pending drafts for a human supervisor to review and approve.

**Scheduled Send Monitoring**: Filter by 'scheduled' label to see all emails queued for future delivery.
<!-- END MANUAL -->

---

## Update Draft

### What it is
A block that updates an existing draft's content, recipients, or scheduled send time.

### How it works
<!-- MANUAL: how_it_works -->
The block modifies a draft's recipients, subject, body, or scheduled send time. Use this to reschedule a draft, edit content before sending, or change recipients. To cancel a scheduled send entirely, delete the draft instead.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to update | str | Yes |
| to | Updated recipient email addresses (replaces existing list) | List[str] | No |
| subject | Updated subject line | str | No |
| text | Updated plain text body | str | No |
| html | Updated HTML body | str | No |
| send_at | Reschedule: new ISO 8601 send time | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| draft_id | The updated draft ID | str |
| send_status | Updated send status | str |
| result | Complete updated draft object | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Draft Editing**: Let a human reviewer edit an AI-generated draft before it's sent.

**Rescheduling**: Change the scheduled send time of a draft based on new information or priorities.
<!-- END MANUAL -->

---

## Send Draft

### What it is
A block that sends a draft immediately, converting it into a delivered message.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the draft and deletes it from the drafts list. The draft becomes a regular message with a message_id and thread_id. Use this as the final step in a human-in-the-loop approval workflow: agent creates draft, human reviews, then this block sends it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to send now | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| message_id | Message ID of the now-sent email (draft is deleted) | str |
| thread_id | Thread ID the sent message belongs to | str |
| result | Complete sent message object | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Approval Workflow**: After a human approves a draft, send it immediately with this block.

**Batch Sending**: Programmatically send multiple reviewed drafts as part of an outreach campaign.
<!-- END MANUAL -->

---

## Delete Draft

### What it is
A block that deletes a draft from an AgentMail inbox, also cancelling any scheduled send.

### How it works
<!-- MANUAL: how_it_works -->
The block permanently removes the draft. If the draft was scheduled with `send_at`, deleting it cancels the scheduled delivery. This is the way to cancel a scheduled email.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the draft belongs to | str | Yes |
| draft_id | Draft ID to delete (also cancels scheduled sends) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| success | True if the draft was successfully deleted/cancelled | bool |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cancel Scheduled Email**: Delete a scheduled draft to prevent it from being sent.

**Draft Cleanup**: Remove outdated or rejected drafts from the inbox.
<!-- END MANUAL -->

---

## List Org Drafts

### What it is
A block that lists all drafts across every inbox in your AgentMail organization.

### How it works
<!-- MANUAL: how_it_works -->
The block returns drafts from all inboxes in one query. Unlike per-inbox listing, this gives a global view of all pending drafts. Supports pagination for organizations with many drafts.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of drafts to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| drafts | List of draft objects from all inboxes in the organization | List[dict] |
| count | Number of drafts returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Central Approval Dashboard**: Build a single view where a human supervisor can review and approve drafts created by any agent across the organization.

**Audit Trail**: Monitor all pending outbound emails across the organization for compliance review.
<!-- END MANUAL -->
