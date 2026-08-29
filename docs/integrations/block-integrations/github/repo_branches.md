# GitHub Repo Branches
<!-- MANUAL: file_description -->
Blocks for creating, listing, deleting, and comparing branches in GitHub repositories.
<!-- END MANUAL -->

## Github Compare Branches

### What it is
This block compares two branches or commits in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block compares two branches or commits using the GitHub Compare API (`/compare/base...head`). It returns the comparison status (ahead, behind, diverged, or identical), commit counts, and a list of changed files with their diffs.

The block also builds a unified diff string from the patches of all changed files, making it easy to review the full set of changes at once.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| base | Base branch or commit SHA | str | Yes |
| head | Head branch or commit SHA to compare against base | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if comparison failed | str |
| status | Comparison status: ahead, behind, diverged, or identical | str |
| ahead_by | Number of commits head is ahead of base | int |
| behind_by | Number of commits head is behind base | int |
| total_commits | Total number of commits in the comparison | int |
| diff | Unified diff of all file changes | str |
| file | A changed file with its diff | Changed File |
| files | List of changed files with their diffs | List[FileChange] |

### Possible use case
<!-- MANUAL: use_case -->
**Pre-Merge Review**: Compare a feature branch against main to see all changes before creating a pull request.

**Drift Detection**: Check whether a long-lived branch has diverged from the base branch and needs rebasing.

**Release Diffing**: Compare two release tags or branches to generate a summary of changes between versions.
<!-- END MANUAL -->

---

## Github Delete Branch

### What it is
This block deletes a specified branch.

### How it works
<!-- MANUAL: how_it_works -->
This block deletes a branch by sending a DELETE request to the GitHub Git Refs API at `/git/refs/heads/{branch}`. The branch name is URL-encoded to handle special characters.

The operation is permanent and cannot be undone. The block returns a success status message or an error if the deletion fails.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| branch | Name of the branch to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the branch deletion failed | str |
| status | Status of the branch deletion operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Post-Merge Cleanup**: Automatically delete feature branches after their pull requests have been merged.

**Stale Branch Removal**: Delete branches that are no longer active to keep the repository organized.

**CI/CD Pipeline Cleanup**: Remove temporary branches created by automated build or test processes.
<!-- END MANUAL -->

---

## Github List Branches

### What it is
This block lists all branches for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves branches from a GitHub repository using the Branches API. It supports pagination with the `per_page` parameter to control how many branches are returned per request.

Each branch entry includes its name and a URL to browse the repository file tree at that branch.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| per_page | Number of branches to return per page (max 100) | int | No |
| page | Page number for pagination | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing branches failed | str |
| branch | Branches with their name and file tree browser URL | Branch |
| branches | List of branches with their name and file tree browser URL | List[BranchItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Branch Inventory**: List all branches to get an overview of active development streams in a repository.

**Stale Branch Detection**: Enumerate branches to identify those that may need cleanup or deletion.

**Branch Validation**: Verify that expected branches exist before running automated workflows.
<!-- END MANUAL -->

---

## Github Make Branch

### What it is
This block creates a new branch from a specified source branch.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new branch by first fetching the latest commit SHA from the source branch via the Git Refs API, then creating a new git reference pointing to that same commit. This effectively branches off from the current tip of the source branch.

The block returns a success status message or an error if the branch creation fails.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| new_branch | Name of the new branch | str | Yes |
| source_branch | Name of the source branch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the branch creation failed | str |
| status | Status of the branch creation operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Feature Branch Creation**: Automatically create feature branches from the main branch when starting new tasks.

**Release Branching**: Create release branches from development branches as part of a release workflow.

**Hotfix Isolation**: Spin up hotfix branches from production branches to isolate urgent fixes.
<!-- END MANUAL -->

---
