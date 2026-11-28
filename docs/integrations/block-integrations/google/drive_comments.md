# Google Drive Comments
<!-- MANUAL: file_description -->
Blocks for reading the comments people leave on Google Docs, Sheets, Slides and other Drive files. Pick the file with the Drive picker, or connect one from Google Drive Search Files. They need read access to Drive (the `drive.readonly` scope).
<!-- END MANUAL -->

## Google Drive List Comments

### What it is
List the comment threads on a Google Doc, Sheet, Slides deck or other Drive file: who said what, the text each comment is on, replies, and whether the thread is resolved.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Drive API `comments.list` endpoint, following pages until it has enough threads. Each thread comes back with its author, text, the passage of the file it's attached to, whether it's resolved, and its replies, oldest first. Deleted replies are left out, and resolved threads can be left out too. Author emails only appear when Google shares them with the connected account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Doc, Sheet, Slides deck or other Drive file whose comments to list. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |
| include_resolved | Include resolved threads | bool | No |
| max_results | Maximum number of threads to return | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comments | Comment threads, oldest first, each with its replies | List[DriveComment] |
| comment | Each comment thread | DriveComment |

### Possible use case
<!-- MANUAL: use_case -->
**Review Digest**: Summarize the open review comments on a document before a meeting.

**Contract Follow-up**: Find unresolved comments across a folder of contracts.

**Revision Drafts**: Feed reviewers' feedback into an AI block that drafts the revisions.
<!-- END MANUAL -->

---
