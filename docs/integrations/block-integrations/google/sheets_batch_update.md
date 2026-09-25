# Google Sheets Batch Update
<!-- MANUAL: file_description -->
A block for power users and the copilot: send any Google Sheets API batchUpdate requests in one call. Use it for changes the other Google Sheets blocks don't cover. Pick the spreadsheet with the Drive picker, or connect one from another Google block.
<!-- END MANUAL -->

## Google Sheets Batch Update

### What it is
Apply any Google Sheets API batchUpdate requests to a spreadsheet in one all-or-nothing call: charts, conditional formatting, merges, filters and anything the other Google Sheets blocks don't cover.

### How it works
<!-- MANUAL: how_it_works -->
Sends your list of requests to the Sheets API `spreadsheets.batchUpdate` endpoint. Google applies them in order and all or nothing: if one request is invalid, nothing changes and the block reports Google's error, which names the request and field to fix. The block returns one reply per request (for example, the ID of a chart it added) and the spreadsheet for chaining.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The Google Sheets spreadsheet to update. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Spreadsheet | No |
| requests | Google Sheets API batchUpdate requests, applied in order and all or nothing, e.g. [{"addChart": {...}}] or [{"repeatCell": {...}}]. Request types: https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/request | List[Dict[str, Any]] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| replies | One reply per request, in order (empty for requests that return nothing) | List[Dict[str, Any]] |
| spreadsheet | The spreadsheet, for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
- Add a chart or conditional formatting to a report sheet.
- Merge header cells, freeze rows and resize columns in one step.
- Let the copilot apply formatting it has worked out from the sheet's metadata.
<!-- END MANUAL -->

---
