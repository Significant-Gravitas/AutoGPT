# Agent Mail Attachments
<!-- MANUAL: file_description -->
Blocks for downloading file attachments from AgentMail messages and threads. Attachments are files associated with messages (PDFs, CSVs, images, etc.) and are returned as base64-encoded content.
<!-- END MANUAL -->

## Agent Mail Get Message Attachment

### What it is
Download a file attachment from an email message. Returns base64-encoded file content.

### How it works
<!-- MANUAL: how_it_works -->
The block calls the AgentMail API's `inboxes.messages.get_attachment` endpoint using the provided inbox ID, message ID, and attachment ID. The API returns the raw file content, which may arrive as `bytes` or `str` depending on the attachment type. The block base64-encodes the result: binary data is encoded directly, while string data is first UTF-8 encoded then base64-encoded. If the API returns an unexpected data type, the block raises a `TypeError`.

On any failure — invalid IDs, network errors, authentication issues, or unexpected response types — the block catches the exception and yields the error message on the `error` output instead of `content_base64`. No partial results are returned; the block either yields both `content_base64` and `attachment_id` on success, or only `error` on failure.
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
**Invoice Processing Pipeline** — Download PDF invoices from incoming messages and feed them into a parsing block that extracts line items and totals.
**Automated Attachment Archival** — Pull attachments from specific senders and store the base64 content in a database or cloud bucket for long-term retention.
**Image Analysis Workflow** — Retrieve image attachments from support emails and pass them to a vision model block for classification or OCR.
<!-- END MANUAL -->

---

## Agent Mail Get Thread Attachment

### What it is
Download a file attachment from a conversation thread. Returns base64-encoded file content.

### How it works
<!-- MANUAL: how_it_works -->
The block calls the AgentMail API's `inboxes.threads.get_attachment` endpoint using the provided inbox ID, thread ID, and attachment ID. This is functionally identical to the message attachment block but resolves the attachment via a thread rather than a specific message — useful when you have the thread context but not the individual message ID. The raw response is base64-encoded the same way: `bytes` are encoded directly, `str` is UTF-8 encoded first, and any other response type triggers a `TypeError`.

Error handling follows the same all-or-nothing pattern: on success the block yields `content_base64` and `attachment_id`; on any exception (bad IDs, auth failure, network error, unexpected type) it yields only `error` with the exception message.
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
**Conversation Attachment Collector** — Iterate over a support thread and download every attachment to build a complete case file for review.
**Threaded Report Extraction** — Pull CSV or Excel attachments from recurring report threads and forward them to a data-processing block.
**Compliance Document Retrieval** — Download signed-document attachments from legal threads and pass them to a verification or archival workflow.
<!-- END MANUAL -->

---
