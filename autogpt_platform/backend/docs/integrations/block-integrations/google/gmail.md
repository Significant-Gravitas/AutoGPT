# Google Gmail
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Gmail Add Label

### What it is
A block that adds a label to a specific email message in Gmail, creating the label if it doesn't exist.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Create Draft

### What it is
Create draft emails in Gmail with automatic HTML detection and proper text formatting. Plain text drafts preserve natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and formatted correctly.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Draft Reply

### What it is
Create draft replies to Gmail threads with automatic HTML detection and proper text formatting. Plain text draft replies maintain natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and formatted correctly.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Forward

### What it is
Forward Gmail messages to other recipients with automatic HTML detection and proper formatting. Preserves original message threading and attachments.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Get Profile

### What it is
Get the authenticated user's Gmail profile details including email address and message statistics.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| profile | Gmail user profile information | Profile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Get Thread

### What it is
A block that retrieves an entire Gmail thread (email conversation) by ID, returning all messages with decoded bodies for reading complete conversations.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail List Labels

### What it is
A block that retrieves all labels (categories) from a Gmail account for organizing and categorizing emails.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | List of labels | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Read

### What it is
A block that retrieves and reads emails from a Gmail account based on search criteria, returning detailed message information including subject, sender, body, and attachments.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Remove Label

### What it is
A block that removes a label from a specific email message in a Gmail account.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Reply

### What it is
Reply to Gmail threads with automatic HTML detection and proper text formatting. Plain text replies maintain natural paragraph flow without 78-character line wrapping. HTML content is automatically detected and sent with correct MIME type.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Gmail Send

### What it is
Send emails via Gmail with automatic HTML detection and proper text formatting. Plain text emails are sent without 78-character line wrapping, preserving natural paragraph flow. HTML emails are automatically detected and sent with correct MIME type.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
