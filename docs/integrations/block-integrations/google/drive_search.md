# Google Drive Search
<!-- MANUAL: file_description -->
Blocks for finding files in Google Drive. They need read access to the user's whole Drive (the `drive.readonly` scope), because they search every file, not just ones picked or created in AutoGPT. Results are Drive files you can connect straight into the other Google Drive, Docs and Sheets blocks.
<!-- END MANUAL -->

## Google Drive List Recent Files

### What it is
List the most recently used, modified or viewed files in Google Drive. Folders are left out.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Drive API `files.list` endpoint, skips folders and trashed files, and sorts by the timestamp you choose: recency (the latest of the file's dates), last modified by anyone, last modified by you, or last viewed by you. Each file comes back with its ID, link, type, owners and dates, plus the credentials needed to pass it to other Google blocks.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_by | Which timestamp decides how recent a file is | "recency" \| "last_modified" \| "last_modified_by_me" \| "last_viewed_by_me" | No |
| max_results | Maximum number of files to return | int | No |
| page_token | Page token from a previous call, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| files | Recent files, newest first | List[DriveFile] |
| file | Each recent file | DriveFile |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
- Show the user the documents they worked on this week.
- Pick up the most recently edited spreadsheet and summarize it with Google Drive Read File.
<!-- END MANUAL -->

---

## Google Drive Search Files

### What it is
Search Google Drive for files by name, content, type, folder or modified date. Returns files you can pass to other Google blocks.

### How it works
<!-- MANUAL: how_it_works -->
Builds a Drive search query from the filters you set (name, full text, file type, folder, modified date and an optional raw Drive query clause) and calls `files.list`. Results come newest first, except when you search file content, where Drive ranks them by relevance. Shared drives are included unless you turn that off. Use `next_page_token` to page through large result sets.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name_contains | Only files whose name contains this text | str | No |
| text_contains | Only files whose name or content contains this text | str | No |
| file_type | Only files of this type | "any" \| "document" \| "spreadsheet" \| "presentation" \| "folder" \| "pdf" \| "image" \| "video" | No |
| folder_id | Only files directly inside this folder (ID or URL; 'root' for My Drive) | str | No |
| modified_after | Only files modified after this time | str (date-time) | No |
| include_shared_drives | Also search shared drives the user belongs to | bool | No |
| include_trashed | Include files in the trash | bool | No |
| custom_query | Extra Drive search clause, ANDed with the filters above (Drive query syntax, e.g. "'me' in owners") | str | No |
| max_results | Maximum number of files to return | int | No |
| page_token | Page token from a previous search, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| files | Matching files, newest first unless searching content | List[DriveFile] |
| file | Each matching file | DriveFile |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
- Find the latest invoice PDF in a folder and download it.
- Look up a spreadsheet by name and pass it to the Google Sheets blocks.
- Find every document that mentions a customer before a renewal call.
<!-- END MANUAL -->

---
