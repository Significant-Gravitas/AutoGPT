# Agent Mail Drafts
<!-- MANUAL: file_description -->
Blocks for creating, reviewing, editing, sending, and deleting email drafts in AgentMail. Drafts enable human-in-the-loop review, scheduled sending, and multi-step email composition workflows.
<!-- END MANUAL -->

## Agent Mail Create Draft

### What it is
Create a draft email for review or scheduled sending. Use send_at for automatic future delivery.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Delete Draft

### What it is
Delete a draft or cancel a scheduled email. Removes the draft permanently.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Draft

### What it is
Retrieve a draft email to review its contents, recipients, and scheduled send status.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Drafts

### What it is
List drafts in an AgentMail inbox. Filter by labels=['scheduled'] to find pending sends.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Org Drafts

### What it is
List all drafts across every inbox in your organization. Use for central approval dashboards.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Send Draft

### What it is
Send a draft immediately, converting it into a delivered message. The draft is deleted after sending.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Update Draft

### What it is
Update a draft's content, recipients, or scheduled send time. Use to reschedule or edit before sending.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| send_at | Reschedule: new ISO 8601 send time (e.g. '2025-01-20T14:00:00Z') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| draft_id | The updated draft ID | str |
| send_status | Updated send status | str |
| result | Complete updated draft object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
