# GitHub Repo Branches
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Compare Branches

### What it is
This block compares two branches or commits in a GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Delete Branch

### What it is
This block deletes a specified branch.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Branches

### What it is
This block lists all branches for a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| branch | Branches with their name and file tree browser URL | Branch |
| branches | List of branches with their name and file tree browser URL | List[BranchItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Make Branch

### What it is
This block creates a new branch from a specified source branch.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
