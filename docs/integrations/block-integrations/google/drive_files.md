# Google Drive Files
<!-- MANUAL: file_description -->
Blocks that work on one Google Drive file: look it up, read it as text, download it, or check who can access it. Pick the file with the Drive picker, or connect a file from Google Drive Search Files. They need read access to Drive (the `drive.readonly` scope).
<!-- END MANUAL -->

## Google Drive Download File

### What it is
Download a file from Google Drive (up to 50 MB). Google Docs, Sheets and Slides are exported to PDF, Office or text formats.

### How it works
<!-- MANUAL: how_it_works -->
Downloads the file through the Drive API. Google Docs, Sheets and Slides can't be downloaded as they are, so they are exported to PDF, the matching Office format (DOCX, XLSX or PPTX), or text (Markdown, CSV or plain text). Other Google types, such as Drawings, only export as PDF. The file keeps its Drive name. In CoPilot it is saved to the workspace; in agents it is passed on as a data URI. Files over 50 MB are refused.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Drive file to download. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |
| export_format | For Google Docs, Sheets and Slides: PDF, the matching Office format (DOCX, XLSX, PPTX), or text (Markdown, CSV, plain text). Other files download as they are. | "pdf" \| "office" \| "text" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content | The downloaded file (a workspace file in CoPilot, a data URI in agents) | str (file) |
| mime_type | MIME type of the downloaded file | str |
| file | The Drive file that was downloaded | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Email Attachments**: Export a Google Doc as a PDF and attach it to an email.

**File Processing**: Pull an image or ZIP file from Drive for further processing.

**Partner Handoff**: Export a Google Sheet as XLSX for a partner who doesn't use Google.
<!-- END MANUAL -->

---

## Google Drive Get File Info

### What it is
Get a Google Drive file's details: name, type, size, owners, created and modified dates, and the folders it is in.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Drive API `files.get` endpoint and returns the file's name, type, size, owners, created and modified dates, parent folders and sharing state. Works for files in My Drive, files shared with the user, and shared drives.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Drive file to look up. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file | The file with its details | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Freshness Check**: Check when a report was last updated before sending it.

**Folder Lookup**: Find which folder a file is in.

**Ownership Check**: See who owns a file before asking for access or moving it.
<!-- END MANUAL -->

---

## Google Drive Get File Permissions

### What it is
List who can access a Google Drive file: users, groups, domains or anyone with the link, and their roles.

### How it works
<!-- MANUAL: how_it_works -->
Lists the file's permissions with the Drive API, following pages until all of them are returned. Each entry shows who has access (a user, group, domain, or anyone with the link) and their role (owner, writer, commenter or reader).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Drive file to check. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| permissions | Everyone who can access the file, with their role | List[DrivePermission] |
| permission | Each permission | DrivePermission |

### Possible use case
<!-- MANUAL: use_case -->
**Public Link Check**: Check whether a file is shared publicly before sending the link.

**Access Audit**: Audit who can edit a sensitive document.

**Offboarding Review**: Confirm a departing teammate no longer has access to key documents.
<!-- END MANUAL -->

---

## Google Drive Read File

### What it is
Read a Google Drive file as text. Works for Google Docs, Sheets (first sheet), Slides, PDFs and plain-text files such as CSV, JSON or Markdown.

### How it works
<!-- MANUAL: how_it_works -->
Looks up the file's type, then turns it into text. Google Docs are exported as Markdown, Google Sheets as CSV (first sheet only; use the Google Sheets blocks for other sheets or ranges) and Google Slides as plain text. PDFs are downloaded and their text is extracted. Text files such as CSV, JSON, XML and Markdown are downloaded and decoded as UTF-8. Other file types raise an error that points to Google Drive Download File. Files over 50 MB are refused.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Drive file to read. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content | The file as text: Markdown for Google Docs, CSV of the first sheet for Google Sheets, plain text for Slides, PDFs and text files | str |
| file | The file that was read | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Document Summaries**: Summarize a Google Doc or PDF the user found in Drive.

**Data Import**: Feed a CSV stored in Drive into an AI block.

**Meeting Prep**: Read the agenda document before a meeting and draft talking points.
<!-- END MANUAL -->

---
