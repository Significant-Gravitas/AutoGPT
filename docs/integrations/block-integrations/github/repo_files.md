# GitHub Repo Files
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Create File

### What it is
This block creates a new file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| file_path | Path where the file should be created | str | Yes |
| content | Content to write to the file | str | Yes |
| branch | Branch where the file should be created | str | No |
| commit_message | Message for the commit | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the file creation failed | str |
| url | URL of the created file | str |
| sha | SHA of the commit | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Get Repository Tree

### What it is
This block lists the entire file tree of a GitHub repository recursively.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Branch name to get the tree from | str | No |
| recursive | Whether to recursively list the entire tree | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if getting tree failed | str |
| entry | A file or directory in the tree | Tree Entry |
| entries | List of all files and directories in the tree | List[TreeEntry] |
| truncated | Whether the tree was truncated due to size | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Read File

### What it is
This block reads the content of a specified file from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| file_path | Path to the file in the repository | str | Yes |
| branch | Branch to read from | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| text_content | Content of the file (decoded as UTF-8 text) | str |
| raw_content | Raw base64-encoded content of the file | str |
| size | The size of the file (in bytes) | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Read Folder

### What it is
This block reads the content of a specified folder from a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| folder_path | Path to the folder in the repository | str | Yes |
| branch | Branch name to read from (defaults to main) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reading the folder failed | str |
| file | Files in the folder | FileEntry |
| dir | Directories in the folder | DirEntry |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Search Code

### What it is
This block searches for code in GitHub repositories.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query (GitHub code search syntax) | str | Yes |
| repo | Restrict search to a repository (owner/repo format, optional) | str | No |
| per_page | Number of results to return (max 100) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if search failed | str |
| result | A code search result | Result |
| results | List of code search results | List[SearchResult] |
| total_count | Total number of matching results | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Update File

### What it is
This block updates an existing file in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| file_path | Path to the file to update | str | Yes |
| content | New content for the file | str | Yes |
| branch | Branch containing the file | str | No |
| commit_message | Message for the commit | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the updated file | str |
| sha | SHA of the commit | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
