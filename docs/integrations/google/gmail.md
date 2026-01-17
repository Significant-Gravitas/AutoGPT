# Google Gmail
<!-- MANUAL: file_description -->
Blocks for reading, sending, and managing emails in Gmail.
<!-- END MANUAL -->

## Gmail Add Label

### What it is
A block that adds a label to a specific email message in Gmail, creating the label if it doesn't exist.

### How it works
<!-- MANUAL: how_it_works -->
The block first checks if the specified label exists in the user's Gmail account. If it doesn't, it creates the label. Then, it adds the label to the specified email message using the message ID.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_id | Message ID to add label to | str | Yes |
| label_name | Label name to add | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Label addition result | GmailLabelResult |

### Possible use case
<!-- MANUAL: use_case -->
Automatically categorizing incoming customer emails based on their content, adding labels like "Urgent," "Feedback," or "Invoice" for easier processing.
<!-- END MANUAL -->

---

## Gmail Create Draft

### What it is
Create draft emails in Gmail with automatic HTML detection and proper text formatting. Plain text drafts preserve natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and formatted correctly.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a draft email in Gmail without sending it. The draft is saved to your Drafts folder where you can review and send it manually. The block automatically detects HTML content or you can explicitly set the content type.

Plain text emails preserve natural formatting without forced line breaks. HTML emails support rich formatting. File attachments are supported by providing file paths.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| to | Recipient email addresses | List[str] | Yes |
| subject | Email subject | str | Yes |
| body | Email body (plain text or HTML) | str | Yes |
| cc | CC recipients | List[str] | No |
| bcc | BCC recipients | List[str] | No |
| content_type | Content type: 'auto' (default - detects HTML), 'plain', or 'html' | "auto" \| "plain" \| "html" | No |
| attachments | Files to attach | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Draft creation result | GmailDraftResult |

### Possible use case
<!-- MANUAL: use_case -->
**Email Review Workflow**: Create draft emails for human review before sending important communications.

**Newsletter Preparation**: Build email drafts with dynamic content that can be finalized before distribution.

**Template Saving**: Save email templates as drafts for quick access and reuse.
<!-- END MANUAL -->

---

## Gmail Draft Reply

### What it is
Create draft replies to Gmail threads with automatic HTML detection and proper text formatting. Plain text draft replies maintain natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and formatted correctly.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a draft reply within an existing email thread. The draft maintains proper threading so your reply appears in the conversation. Use replyAll to respond to all original recipients, or specify custom recipients.

The block preserves the thread context and adds proper email headers for threading. Draft replies can be reviewed in Gmail before sending.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| threadId | Thread ID to reply in | str | Yes |
| parentMessageId | ID of the message being replied to | str | Yes |
| to | To recipients | List[str] | No |
| cc | CC recipients | List[str] | No |
| bcc | BCC recipients | List[str] | No |
| replyAll | Reply to all original recipients | bool | No |
| subject | Email subject | str | No |
| body | Email body (plain text or HTML) | str | Yes |
| content_type | Content type: 'auto' (default - detects HTML), 'plain', or 'html' | "auto" \| "plain" \| "html" | No |
| attachments | Files to attach | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| draftId | Created draft ID | str |
| messageId | Draft message ID | str |
| threadId | Thread ID | str |
| status | Draft creation status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Support Response Preparation**: Draft replies to customer inquiries for review before sending.

**Approval Workflows**: Create reply drafts that require manager approval before being sent.

**Scheduled Responses**: Prepare replies to be reviewed and sent at appropriate times.
<!-- END MANUAL -->

---

## Gmail Forward

### What it is
Forward Gmail messages to other recipients with automatic HTML detection and proper formatting. Preserves original message threading and attachments.

### How it works
<!-- MANUAL: how_it_works -->
This block forwards an existing Gmail message to new recipients. The original message content is preserved and can include attachments from the original email. You can add your own message before the forwarded content.

The block handles proper email threading and formatting, prepending "Fwd:" to the subject unless you specify a custom subject.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| messageId | ID of the message to forward | str | Yes |
| to | Recipients to forward the message to | List[str] | Yes |
| cc | CC recipients | List[str] | No |
| bcc | BCC recipients | List[str] | No |
| subject | Optional custom subject (defaults to 'Fwd: [original subject]') | str | No |
| forwardMessage | Optional message to include before the forwarded content | str | No |
| includeAttachments | Include attachments from the original message | bool | No |
| content_type | Content type: 'auto' (default - detects HTML), 'plain', or 'html' | "auto" \| "plain" \| "html" | No |
| additionalAttachments | Additional files to attach | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| messageId | Forwarded message ID | str |
| threadId | Thread ID | str |
| status | Forward status | str |

### Possible use case
<!-- MANUAL: use_case -->
**Email Escalation**: Automatically forward emails matching certain criteria to managers or specialists.

**Team Distribution**: Forward important updates to relevant team members based on content.

**Record Keeping**: Forward copies of important communications to an archive address.
<!-- END MANUAL -->

---

## Gmail Get Profile

### What it is
Get the authenticated user's Gmail profile details including email address and message statistics.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves profile information for the authenticated Gmail user via the Gmail API. It returns the email address, total message count, thread count, and storage usage statistics.

This is useful for verifying which account is connected and gathering basic mailbox statistics.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| profile | Gmail user profile information | Profile |

### Possible use case
<!-- MANUAL: use_case -->
**Account Verification**: Confirm which Gmail account is connected before performing operations.

**Usage Monitoring**: Check storage usage and message counts for mailbox management.

**Multi-Account Workflows**: Get the current user's email address to route workflows appropriately.
<!-- END MANUAL -->

---

## Gmail Get Thread

### What it is
A block that retrieves an entire Gmail thread (email conversation) by ID, returning all messages with decoded bodies for reading complete conversations.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a complete Gmail thread (email conversation) by its thread ID. It returns all messages in the thread with decoded bodies, allowing you to read the full conversation history.

The thread includes all messages, their senders, timestamps, and content, making it easy to analyze entire email conversations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| threadId | Gmail thread ID | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| thread | Gmail thread with decoded message bodies | Thread |

### Possible use case
<!-- MANUAL: use_case -->
**Conversation Analysis**: Read an entire email thread to understand the full context of a discussion.

**Reply Detection**: Check if a recipient has responded within a conversation thread.

**Thread Summarization**: Gather all messages in a thread for AI-powered summarization.
<!-- END MANUAL -->

---

## Gmail List Labels

### What it is
A block that retrieves all labels (categories) from a Gmail account for organizing and categorizing emails.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to the user's Gmail account and requests a list of all labels. It then processes this information and returns a simplified list of label names and their corresponding IDs.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | List of labels | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
Creating a dashboard that shows an overview of how many emails are in each category or label in a business email account.
<!-- END MANUAL -->

---

## Gmail Read

### What it is
A block that retrieves and reads emails from a Gmail account based on search criteria, returning detailed message information including subject, sender, body, and attachments.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to the user's Gmail account using their credentials, performs a search based on the provided query, and retrieves the specified number of email messages. It then processes each email to extract relevant information and returns the results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query for reading emails | str | No |
| max_results | Maximum number of emails to retrieve | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| email | Email data | Email |
| emails | List of email data | List[Email] |

### Possible use case
<!-- MANUAL: use_case -->
Automatically checking for new customer inquiries in a support email inbox and organizing them for quick response.
<!-- END MANUAL -->

---

## Gmail Remove Label

### What it is
A block that removes a label from a specific email message in a Gmail account.

### How it works
<!-- MANUAL: how_it_works -->
The block first finds the ID of the specified label in the user's Gmail account. If the label exists, it removes it from the specified email message using the message ID.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| message_id | Message ID to remove label from | str | Yes |
| label_name | Label name to remove | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Label removal result | GmailLabelResult |

### Possible use case
<!-- MANUAL: use_case -->
Automatically removing the "Unread" label from emails after they have been processed by a customer service representative.
<!-- END MANUAL -->

---

## Gmail Reply

### What it is
Reply to Gmail threads with automatic HTML detection and proper text formatting. Plain text replies maintain natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and sent with correct MIME type.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a reply directly within an existing Gmail thread. Unlike the draft reply block, this immediately sends the message. The reply maintains proper threading and appears in the conversation.

Use replyAll to respond to all recipients, or specify custom recipients. The block handles email headers and threading automatically.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| threadId | Thread ID to reply in | str | Yes |
| parentMessageId | ID of the message being replied to | str | Yes |
| to | To recipients | List[str] | No |
| cc | CC recipients | List[str] | No |
| bcc | BCC recipients | List[str] | No |
| replyAll | Reply to all original recipients | bool | No |
| subject | Email subject | str | No |
| body | Email body (plain text or HTML) | str | Yes |
| content_type | Content type: 'auto' (default - detects HTML), 'plain', or 'html' | "auto" \| "plain" \| "html" | No |
| attachments | Files to attach | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| messageId | Sent message ID | str |
| threadId | Thread ID | str |
| message | Raw Gmail message object | Dict[str, Any] |
| email | Parsed email object with decoded body and attachments | Email |

### Possible use case
<!-- MANUAL: use_case -->
**Auto-Acknowledgments**: Automatically send acknowledgment replies to incoming support requests.

**Scheduled Follow-ups**: Reply to threads with follow-up messages at appropriate times.

**Conversation Continuity**: Respond to ongoing threads while keeping all messages organized.
<!-- END MANUAL -->

---

## Gmail Send

### What it is
Send emails via Gmail with automatic HTML detection and proper text formatting. Plain text emails are sent without 78-character line wrapping, preserving natural paragraph flow. HTML emails are automatically detected and sent with correct MIME type.

### How it works
<!-- MANUAL: how_it_works -->
The block authenticates with the user's Gmail account, creates an email message with the provided details (recipient, subject, and body), and then sends the email using Gmail's API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| to | Recipient email addresses | List[str] | Yes |
| subject | Email subject | str | Yes |
| body | Email body (plain text or HTML) | str | Yes |
| cc | CC recipients | List[str] | No |
| bcc | BCC recipients | List[str] | No |
| content_type | Content type: 'auto' (default - detects HTML), 'plain', or 'html' | "auto" \| "plain" \| "html" | No |
| attachments | Files to attach | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Send confirmation | GmailSendResult |

### Possible use case
<!-- MANUAL: use_case -->
Automatically sending confirmation emails to customers after they make a purchase on an e-commerce website.
<!-- END MANUAL -->

---
