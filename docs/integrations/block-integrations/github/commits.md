# GitHub Commits
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github List Commits

### What it is
This block lists commits on a branch in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Branch name to list commits from | str | No |
| per_page | Number of commits to return (max 100) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing commits failed | str |
| commit | A commit with its details | Commit |
| commits | List of commits with their details | List[CommitItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Multi File Commit

### What it is
This block creates a single commit with multiple file upsert/delete operations using the Git Trees API.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Branch to commit to | str | Yes |
| commit_message | Commit message | str | Yes |
| files | List of file operations. Each item has: 'path' (file path), 'content' (file content, ignored for delete), 'operation' (upsert/delete) | List[FileOperationInput] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the commit failed | str |
| sha | SHA of the new commit | str |
| url | URL of the new commit | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
