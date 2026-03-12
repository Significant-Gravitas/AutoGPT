# GitHub Commits
<!-- MANUAL: file_description -->
Blocks for listing commit history and creating multi-file commits in GitHub repositories.
<!-- END MANUAL -->

## Github List Commits

### What it is
This block lists commits on a branch in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block fetches commits from a GitHub repository using the Commits API. You can filter by branch name and control the number of results with pagination parameters.

Each commit entry includes the SHA, commit message, author name, date, and a URL to view the commit on GitHub.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Branch name to list commits from | str | No |
| per_page | Number of commits to return (max 100) | int | No |
| page | Page number for pagination | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing commits failed | str |
| commit | A commit with its details | Commit |
| commits | List of commits with their details | List[CommitItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Activity Monitoring**: Track recent commits on a branch to monitor development progress.

**Changelog Generation**: Retrieve commit messages to automatically compile release notes or changelogs.

**Audit Trail**: List commits to review who made changes and when for compliance purposes.
<!-- END MANUAL -->

---

## Github Multi File Commit

### What it is
This block creates a single commit with multiple file upsert/delete operations using the Git Trees API.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a single atomic commit that can add, update, or delete multiple files at once using the low-level Git Trees API. It fetches the latest commit SHA on the target branch, creates blobs for each upserted file concurrently, builds a new tree with all file operations, creates a commit pointing to that tree, and updates the branch reference.

For delete operations, a null SHA is set in the tree entry to remove the file. This approach is more efficient than making separate commits per file and ensures all changes land in a single commit.
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
**Automated Code Generation**: Commit generated code, configuration files, and scaffolding in a single atomic operation.

**Batch File Updates**: Update multiple configuration or documentation files across a repository in one commit.

**Cleanup Operations**: Delete obsolete files while adding replacements in a single commit to keep history clean.
<!-- END MANUAL -->

---
