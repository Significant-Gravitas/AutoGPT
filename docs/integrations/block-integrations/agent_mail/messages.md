# Agent Mail Messages
<!-- MANUAL: file_description -->
Blocks for sending, receiving, replying to, forwarding, and managing email messages via AgentMail. Messages are individual emails within conversation threads.
<!-- END MANUAL -->

## Agent Mail Forward Message

### What it is
Forward an email message to one or more recipients. Supports CC/BCC and optional extra text or subject override.

### How it works
<!-- MANUAL: how_it_works -->
The block validates that the combined recipient count across to, cc, and bcc does not exceed 50, then calls the AgentMail API to forward a specific message from your inbox. You provide the inbox ID and message ID to identify the original email, along with the target email addresses. Optionally, you can override the subject line or prepend additional plain text or HTML content before the forwarded message body.

The API handles constructing the forwarded email with the original content included. Any errors from the API propagate directly to the global error handler without being caught by the block.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to forward from | str | Yes |
| message_id | Message ID to forward | str | Yes |
| to | Recipient email addresses to forward the message to (e.g. ['user@example.com']) | List[str] | Yes |
| cc | CC recipient email addresses | List[str] | No |
| bcc | BCC recipient email addresses (hidden from other recipients) | List[str] | No |
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
- **Escalation Routing** — Forward messages that match certain keywords or priority levels to a human supervisor's email for review.
- **Multi-Agent Collaboration** — Forward incoming requests to a specialized agent's inbox so the right agent handles each task.
- **Digest Distribution** — Forward summarized daily reports from an aggregation inbox to a distribution list of stakeholders.
<!-- END MANUAL -->

---

## Agent Mail Get Message

### What it is
Retrieve a specific email message by ID. Includes extracted_text for clean reply content without quoted history.

### How it works
<!-- MANUAL: how_it_works -->
The block fetches a single message from an AgentMail inbox by calling the API with the inbox ID and message ID. It returns the full message content including subject, plain text body, HTML body, and all metadata.

A key output is `extracted_text`, which contains only the new reply content with quoted history stripped out. This is especially useful when feeding message content into an LLM, since it avoids processing redundant quoted text from earlier in the conversation.
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
- **Intent Classification** — Retrieve a message and pass its extracted text to an LLM to classify the sender's intent before routing to the appropriate workflow.
- **Conversation Context Loading** — Fetch a specific message to build context for generating a relevant reply in a multi-turn email conversation.
- **Attachment Processing** — Retrieve a message's full metadata to extract attachment URLs for downstream processing like document parsing or image analysis.
<!-- END MANUAL -->

---

## Agent Mail List Messages

### What it is
List messages in an AgentMail inbox. Filter by labels to find unread, campaign-tagged, or categorized messages.

### How it works
<!-- MANUAL: how_it_works -->
The block queries the AgentMail API to retrieve a paginated list of messages from the specified inbox. You can control the page size with `limit` (1-100) and navigate through results using the `page_token` returned from a previous call. An optional `labels` filter returns only messages that have all of the specified labels.

The block outputs the list of message objects, a count of messages returned in the current page, and a `next_page_token` for fetching subsequent pages. When `next_page_token` is empty, there are no more results.
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
- **Inbox Polling** — Periodically list messages labeled "unread" to trigger automated processing workflows for new incoming emails.
- **Campaign Monitoring** — Filter messages by campaign-specific labels to track reply rates and engagement across an outreach sequence.
- **Batch Processing** — Page through all messages in an inbox to perform bulk operations like summarization, archiving, or data extraction.
<!-- END MANUAL -->

---

## Agent Mail Reply To Message

### What it is
Reply to an existing email in the same conversation thread. Use for multi-turn agent conversations.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a reply to an existing message by calling the AgentMail API with the inbox ID, the message ID being replied to, and the reply body text. An optional HTML body can be provided for rich formatting. The API automatically threads the reply into the same conversation as the original message.

The block returns the new reply's message ID, the thread ID it was added to, and the complete message object. This makes it straightforward to build multi-turn email conversations where an agent responds to incoming messages within the same thread.
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
- **Customer Support Agent** — Automatically reply to incoming support emails with answers generated by an LLM based on the message content and a knowledge base.
- **Interview Scheduling** — Reply to candidate emails with proposed interview times after checking calendar availability through another block.
- **Conversational Workflow** — Maintain an ongoing back-and-forth conversation with a user, where each reply builds on the previous exchange to complete a multi-step task.
<!-- END MANUAL -->

---

## Agent Mail Send Message

### What it is
Send a new email from an AgentMail inbox. Creates a new conversation thread. Supports HTML, CC/BCC, and labels.

### How it works
<!-- MANUAL: how_it_works -->
The block first validates that the combined count of `to`, `cc`, and `bcc` recipients does not exceed 50. It then calls the AgentMail API to send a new message from the specified inbox, creating a new conversation thread. You must provide at least a plain text body; an optional HTML body can be included for rich formatting.

The block supports CC and BCC recipients for human-in-the-loop oversight or silent monitoring, and labels for tagging outgoing messages for later filtering. The API returns the new message's ID, the thread ID for tracking future replies, and the complete message object.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address to send from (e.g. 'agent@agentmail.to') | str | Yes |
| to | Recipient email addresses (e.g. ['user@example.com']) | List[str] | Yes |
| subject | Email subject line | str | Yes |
| text | Plain text body of the email. Always provide this as a fallback for email clients that don't render HTML. | str | Yes |
| html | Rich HTML body of the email. Embed CSS in a <style> tag for best compatibility across email clients. | str | No |
| cc | CC recipient email addresses for human-in-the-loop oversight | List[str] | No |
| bcc | BCC recipient email addresses (hidden from other recipients) | List[str] | No |
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
- **Outreach Campaigns** — Send personalized cold emails to a list of prospects with campaign labels for tracking, using HTML templates for professional formatting.
- **Alert Notifications** — Send automated alert emails when a monitored metric crosses a threshold, CC-ing a human operator for oversight.
- **Report Delivery** — Generate and send periodic summary reports to stakeholders with BCC to an archive inbox for record-keeping.
<!-- END MANUAL -->

---

## Agent Mail Update Message

### What it is
Add or remove labels on an email message. Use for read/unread tracking, campaign tagging, or state management.

### How it works
<!-- MANUAL: how_it_works -->
The block calls the AgentMail API to modify the labels on a specific message. You can add new labels, remove existing ones, or do both in a single call. Labels are arbitrary strings you define, making them flexible for tracking message state such as read/unread, processing status, or campaign membership.

The block returns the updated message ID and the complete message object reflecting the current label state. This is useful as a downstream step after processing a message, allowing you to mark it as handled so it is not picked up again by a polling workflow.
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
- **Read/Unread Tracking** — Remove the "unread" label and add "read" after an agent processes a message, preventing duplicate processing on the next polling cycle.
- **Pipeline State Management** — Add labels like "sentiment-analyzed" or "response-drafted" as a message moves through multi-step processing, so each stage knows which messages still need work.
- **Priority Tagging** — Add a "high-priority" label to messages from VIP senders or containing urgent keywords, enabling downstream blocks to filter and handle them first.
<!-- END MANUAL -->

---
