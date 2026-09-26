# Google Docs Batch Update
<!-- MANUAL: file_description -->
A block for power users and the copilot: send any Google Docs API batchUpdate requests in one call. Use it for edits the other Google Docs blocks don't cover. Pick the document with the Drive picker, or connect one from another Google block.
<!-- END MANUAL -->

## Google Docs Batch Update

### What it is
Apply any Google Docs API batchUpdate requests to a document in one all-or-nothing call: named ranges, bullets, headers, footnotes, images and anything the other Google Docs blocks don't cover.

### How it works
<!-- MANUAL: how_it_works -->
Sends your list of requests to the Docs API `documents.batchUpdate` endpoint. Google applies them in order and all or nothing: if one request is invalid, nothing changes and the block reports Google's error, which names the request and field to fix. Set `required_revision_id` to a revision from an earlier read or update to make the update fail instead of overwriting someone else's newer edit. The block returns one reply per request (for example, the ID of a created named range) and the document's new revision.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | The Google Doc to update. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Document | No |
| requests | Google Docs API batchUpdate requests, applied in order and all or nothing, e.g. [{"insertText": {"location": {"index": 1}, "text": "Hello\n"}}]. Request types: https://developers.google.com/workspace/docs/api/reference/rest/v1/documents/request | List[Dict[str, Any]] | Yes |
| required_revision_id | Only apply the update if the document is still at this revision (from an earlier read or update). Empty always applies it. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| replies | One reply per request, in order (empty for requests that return nothing) | List[Dict[str, Any]] |
| revision_id | The document's revision after the update | str |
| document | The document, for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Document Structure**: Add bullets, headers, footers or footnotes in one step.

**Named Sections**: Create named ranges so later updates can find a section.

**Precise AI Edits**: Let the copilot make structural edits it has planned from the document's structure.
<!-- END MANUAL -->

---
