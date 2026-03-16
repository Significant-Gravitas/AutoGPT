# AgentMail Messages
<!-- MANUAL: file_description -->
Blocks for sending, receiving, replying to, forwarding, and managing email messages via AgentMail. Messages are individual emails within conversation threads.
<!-- END MANUAL -->

## Send Message

### What it is
A block that sends a new email from an AgentMail inbox, automatically creating a new conversation thread.

### How it works
<!-- MANUAL: how_it_works -->
The block sends an email from the specified inbox to the given recipient. It supports plain text and HTML bodies, CC/BCC recipients, and labels for organizing messages. A new thread is automatically created for each sent message. Max 50 combined recipients across to, cc, and bcc.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to send from | str | Yes |
| to | Recipient email address | str | Yes |
| subject | Email subject line | str | Yes |
| text | Plain text body of the email | str | Yes |
| html | Rich HTML body of the email | str | No |
| cc | CC recipient email address | str | No |
| bcc | BCC recipient email address (hidden from other recipients) | str | No |
| labels | Labels to tag the message for filtering and state management | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| message_id | Unique identifier of the sent message | str |
| thread_id | Thread ID grouping this message and any future replies | str |
| result | Complete sent message object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Outreach**: Send personalized emails to leads or customers as part of an AI-driven sales or support workflow.

**Notification System**: Have an AI agent send transactional emails like order confirmations, status updates, or alerts.
<!-- END MANUAL -->

---

## List Messages

### What it is
A block that lists all messages in an AgentMail inbox with optional label filtering.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves a paginated list of messages from the specified inbox. Use labels to filter results (e.g. labels=['unread'] to only get unprocessed messages). Supports pagination via page tokens for large inboxes.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to list messages from | str | Yes |
| limit | Maximum number of messages to return per page (1-100) | int | No |
| page_token | Token from a previous response to fetch the next page | str | No |
| labels | Only return messages with ALL of these labels | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| messages | List of message objects with subject, sender, text, html, labels, etc. | List[dict] |
| count | Number of messages returned | int |
| next_page_token | Token for the next page. Empty if no more results. | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inbox Polling**: Periodically check for new messages labeled 'unread' and route them to the appropriate AI agent for processing.

**Campaign Monitoring**: List all messages tagged with a specific campaign label to track engagement.
<!-- END MANUAL -->

---

## Get Message

### What it is
A block that retrieves a specific email message by ID from an AgentMail inbox.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches the full message including subject, body (text and HTML), sender, recipients, and attachments. The `extracted_text` output provides only the new reply content without quoted history, which is ideal for AI processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the message belongs to | str | Yes |
| message_id | Message ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| message_id | Unique identifier of the message | str |
| thread_id | Thread this message belongs to | str |
| subject | Email subject line | str |
| text | Full plain text body (may include quoted reply history) | str |
| extracted_text | Just the new reply content with quoted history stripped | str |
| html | HTML body of the email | str |
| result | Complete message object with all fields including sender, recipients, attachments, labels | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Reply Processing**: Fetch a message and use `extracted_text` to get only the new content for AI analysis, ignoring quoted history.

**Email Triage**: Retrieve full message details to classify and route incoming emails to the right workflow.
<!-- END MANUAL -->

---

## Reply To Message

### What it is
A block that replies to an existing email message, keeping the reply in the same conversation thread.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a reply to a specific message, automatically adding it to the same conversation thread. Supports plain text and HTML bodies. Use this for multi-turn agent conversations where context continuity matters.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to send the reply from | str | Yes |
| message_id | Message ID to reply to | str | Yes |
| text | Plain text body of the reply | str | Yes |
| html | Rich HTML body of the reply | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| message_id | Unique identifier of the reply message | str |
| thread_id | Thread ID the reply was added to | str |
| result | Complete reply message object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Support Agent**: Automatically reply to customer inquiries with AI-generated responses, keeping the full conversation in one thread.

**Follow-up Automation**: Send automated follow-up replies based on the content of incoming messages.
<!-- END MANUAL -->

---

## Forward Message

### What it is
A block that forwards an existing email message to a new recipient.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the original message content to a different email address. You can optionally prepend additional text or HTML, and override the subject line. If no subject is provided, it defaults to "Fwd: <original subject>".
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to forward from | str | Yes |
| message_id | Message ID to forward | str | Yes |
| to | Email address to forward the message to | str | Yes |
| subject | Override the subject line | str | No |
| text | Additional plain text to prepend before the forwarded content | str | No |
| html | Additional HTML to prepend before the forwarded content | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| message_id | Unique identifier of the forwarded message | str |
| thread_id | Thread ID of the forward | str |
| result | Complete forwarded message object with all metadata | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Escalation Workflow**: Forward complex customer issues to a human team member with additional context prepended by the AI agent.

**Cross-Team Routing**: Automatically forward emails matching certain criteria to the appropriate department.
<!-- END MANUAL -->

---

## Update Message

### What it is
A block that adds or removes labels on an email message for state management.

### How it works
<!-- MANUAL: how_it_works -->
The block modifies the labels on a message. Labels are string tags used to track message state (read/unread), categorize messages (billing, support), or tag campaigns (q4-outreach). A common pattern is to add 'read' and remove 'unread' after processing a message.
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
| message_id | The updated message ID | str |
| result | Complete updated message object with current labels | dict |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Read/Unread Tracking**: Mark messages as 'read' after an AI agent processes them, so they aren't processed again.

**Campaign Tagging**: Label messages with campaign identifiers for analytics and reporting.
<!-- END MANUAL -->
