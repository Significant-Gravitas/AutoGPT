# Agent Mail Messages
<!-- MANUAL: file_description -->
Blocks for sending, receiving, replying to, forwarding, and managing email messages via AgentMail. Messages are individual emails within conversation threads.
<!-- END MANUAL -->

## Agent Mail Forward Message

### What it is
Forward an email message to a new recipient. Optionally add extra text or change the subject.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to forward from | str | Yes |
| message_id | Message ID to forward | str | Yes |
| to | Email address to forward the message to | str | Yes |
| subject | Override the subject line (defaults to 'Fwd: <original subject>') | str | No |
| text | Additional plain text to prepend before the forwarded content | str | No |
| html | Additional HTML to prepend before the forwarded content | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | Unique identifier of the forwarded message | str |
| thread_id | Thread ID of the forward | str |
| result | Complete forwarded message object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Message

### What it is
Retrieve a specific email message by ID. Includes extracted_text for clean reply content without quoted history.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the message belongs to | str | Yes |
| message_id | Message ID to retrieve (e.g. '<abc123@agentmail.to>') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | Unique identifier of the message | str |
| thread_id | Thread this message belongs to | str |
| subject | Email subject line | str |
| text | Full plain text body (may include quoted reply history) | str |
| extracted_text | Just the new reply content with quoted history stripped. Best for AI processing. | str |
| html | HTML body of the email | str |
| result | Complete message object with all fields including sender, recipients, attachments, labels | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail List Messages

### What it is
List messages in an AgentMail inbox. Filter by labels to find unread, campaign-tagged, or categorized messages.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to list messages from | str | Yes |
| limit | Maximum number of messages to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Only return messages with ALL of these labels (e.g. ['unread'] or ['q4-campaign', 'follow-up']) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| messages | List of message objects with subject, sender, text, html, labels, etc. | List[Dict[str, Any]] |
| count | Number of messages returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Reply To Message

### What it is
Reply to an existing email in the same conversation thread. Use for multi-turn agent conversations.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to send the reply from | str | Yes |
| message_id | Message ID to reply to (e.g. '<abc123@agentmail.to>') | str | Yes |
| text | Plain text body of the reply | str | Yes |
| html | Rich HTML body of the reply | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | Unique identifier of the reply message | str |
| thread_id | Thread ID the reply was added to | str |
| result | Complete reply message object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Send Message

### What it is
Send a new email from an AgentMail inbox. Creates a new conversation thread. Supports HTML, CC/BCC, and labels.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to send from (e.g. 'agent@agentmail.to') | str | Yes |
| to | Recipient email address (e.g. 'user@example.com') | str | Yes |
| subject | Email subject line | str | Yes |
| text | Plain text body of the email. Always provide this as a fallback for email clients that don't render HTML. | str | Yes |
| html | Rich HTML body of the email. Embed CSS in a <style> tag for best compatibility across email clients. | str | No |
| cc | CC recipient email address for human-in-the-loop oversight | str | No |
| bcc | BCC recipient email address (hidden from other recipients) | str | No |
| labels | Labels to tag the message for filtering and state management (e.g. ['outreach', 'q4-campaign']) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | Unique identifier of the sent message | str |
| thread_id | Thread ID grouping this message and any future replies | str |
| result | Complete sent message object with all metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Update Message

### What it is
Add or remove labels on an email message. Use for read/unread tracking, campaign tagging, or state management.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the message belongs to | str | Yes |
| message_id | Message ID to update labels on | str | Yes |
| add_labels | Labels to add (e.g. ['read', 'processed', 'high-priority']) | List[str] | No |
| remove_labels | Labels to remove (e.g. ['unread', 'pending']) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | The updated message ID | str |
| result | Complete updated message object with current labels | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
