# AgentMail Attachments
<!-- MANUAL: file_description -->
Blocks for downloading file attachments from AgentMail messages and threads. Attachments are files associated with messages (PDFs, CSVs, images, etc.) and are returned as base64-encoded content.
<!-- END MANUAL -->

## Get Message Attachment

### What it is
A block that downloads a file attachment from a specific email message.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves the raw file content from a message attachment and returns it as base64-encoded data. First get the attachment_id from a message object's attachments array (via Get Message), then use this block to download the file content.
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
| content_base64 | File content encoded as a base64 string | str |
| attachment_id | The attachment ID that was downloaded | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Document Processing**: Download PDF or CSV attachments from incoming emails for automated data extraction.

**File Archival**: Retrieve and store email attachments in a separate storage system for record-keeping.
<!-- END MANUAL -->

---

## Get Thread Attachment

### What it is
A block that downloads a file attachment from a conversation thread.

### How it works
<!-- MANUAL: how_it_works -->
Same as Get Message Attachment but looks up by thread ID instead of message ID. The block retrieves the raw file content and returns it as base64-encoded data. Useful when you know the thread but not the specific message containing the attachment.
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
| content_base64 | File content encoded as a base64 string | str |
| attachment_id | The attachment ID that was downloaded | str |
| error | Error message if the operation failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Thread-Based File Retrieval**: Download attachments from a conversation thread when you don't have the specific message ID.

**Invoice Processing**: Retrieve invoice attachments from support threads for automated accounting workflows.
<!-- END MANUAL -->
