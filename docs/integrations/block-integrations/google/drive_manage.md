# Google Drive Manage
<!-- MANUAL: file_description -->
Blocks that change Google Drive: create files and folders, copy files, and move them between folders. Creating only needs the narrow `drive.file` scope, which covers files AutoGPT creates. Copying and moving work on any file the user can edit, so they ask for full Drive access (the `drive` scope).
<!-- END MANUAL -->

## Google Drive Copy File

### What it is
Copy a Google Drive file, optionally with a new name or into another folder.

### How it works
<!-- MANUAL: how_it_works -->
Copies the file with the Drive API `files.copy` endpoint. Without a new name the copy is called "Copy of <name>"; without a folder it goes next to the original.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Drive file to copy. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |
| new_name | Name for the copy. Empty means 'Copy of <name>'. | str | No |
| folder_id | Folder for the copy (ID or URL). Empty means the original's folder. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file | The copy | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Templates**: Copy a template document before filling it in.

**Project Setup**: Duplicate a spreadsheet for each new project.

**Safe Edits**: Make a backup copy of a file before an agent edits it.
<!-- END MANUAL -->

---

## Google Drive Create File

### What it is
Create a file in Google Drive from text or an uploaded file, optionally converting it to a Google Doc, Sheet or Slides file.

### How it works
<!-- MANUAL: how_it_works -->
Uploads text or a file with the Drive API `files.create` endpoint. The MIME type comes from the file name unless you set one. Set `convert_to` to turn the content into a Google Doc, Sheet or Slides file (for example, a CSV into a Google Sheet); with no content, it creates an empty one. Without a folder, the file goes in My Drive.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | Name of the new file, e.g. notes.txt. Defaults to the uploaded file's name. | str | No |
| text_content | Text to put in the file. Leave empty when uploading a file. | str | No |
| file_to_upload | A file to upload instead of text (URL, data URI or workspace file) | str (file) | No |
| convert_to | Convert the content into a Google Doc, Sheet or Slides file. With no content, creates an empty one. | "none" \| "google_doc" \| "google_sheet" \| "google_slides" | No |
| folder_id | Folder to create the file in (ID or URL). Empty means My Drive. | str | No |
| content_type | MIME type of the content. Worked out from the name when empty. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file | The new file | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Save Reports**: Save an AI-written report to Drive as a Google Doc.

**Share Data**: Upload a generated CSV and convert it to a Google Sheet for the team.

**Archive Outputs**: Store files an agent produced, such as images or PDFs, in a project folder.
<!-- END MANUAL -->

---

## Google Drive Create Folder

### What it is
Create a folder in Google Drive, optionally inside another folder.

### How it works
<!-- MANUAL: how_it_works -->
Creates a folder with the Drive API, inside the parent folder you give or in My Drive.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | Name of the new folder | str | Yes |
| parent_folder_id | Folder to create it in (ID or URL). Empty means My Drive. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| folder | The new folder | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Client Folders**: Create a folder per client before saving files into it.

**Monthly Archives**: Create a folder for each month's invoices.

**Project Setup**: Create a project folder when a new deal closes.
<!-- END MANUAL -->

---

## Google Drive Move File

### What it is
Move a Google Drive file into another folder.

### How it works
<!-- MANUAL: how_it_works -->
Reads the file's current parent folders, then updates the file to add the destination folder and remove the old ones. Accepts a folder ID, a folder URL, or `root` for My Drive.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | The Drive file to move. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | File | No |
| destination_folder_id | Folder to move it to (ID or URL; 'root' for My Drive) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file | The file in its new folder | DriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Invoice Filing**: File processed invoices into an archive folder.

**Upload Sorting**: Sort new uploads into per-client folders.

**Cleanup**: Move finished drafts out of a shared inbox folder.
<!-- END MANUAL -->

---
