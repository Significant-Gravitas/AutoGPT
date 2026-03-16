# Agent Mail Attachments
<!-- MANUAL: file_description -->
Blocks for downloading file attachments from AgentMail messages and threads. Attachments are files associated with messages (PDFs, CSVs, images, etc.) and are returned as base64-encoded content.
<!-- END MANUAL -->

## Agent Mail Get Message Attachment

### What it is
Download a file attachment from an email message. Returns base64-encoded file content.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the message belongs to | str | Yes |
| message_id | Message ID containing the attachment | str | Yes |
| attachment_id | Attachment ID to download (from the message's attachments array) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content_base64 | File content encoded as a base64 string. Decode with base64.b64decode() to get raw bytes. | str |
| attachment_id | The attachment ID that was downloaded | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Mail Get Thread Attachment

### What it is
Download a file attachment from a conversation thread. Returns base64-encoded file content.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_id | Inbox ID or email address the thread belongs to | str | Yes |
| thread_id | Thread ID containing the attachment | str | Yes |
| attachment_id | Attachment ID to download (from a message's attachments array within the thread) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content_base64 | File content encoded as a base64 string. Decode with base64.b64decode() to get raw bytes. | str |
| attachment_id | The attachment ID that was downloaded | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
