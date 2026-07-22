# GitHub Pull Requests
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Assign PR Reviewer

### What it is
This block assigns a reviewer to a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| reviewer | Username of the reviewer to assign | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the reviewer assignment failed | str |
| status | Status of the reviewer assignment operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List PR Reviewers

### What it is
This block lists all reviewers for a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if listing reviewers failed | str |
| reviewer | Reviewers with their username and profile URL | Reviewer |
| reviewers | List of reviewers with their username and profile URL | List[ReviewerItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Pull Requests

### What it is
This block lists all pull requests for a specified GitHub repository.

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
| error | Error message if listing pull requests failed | str |
| pull_request | PRs with their title and URL | Pull Request |
| pull_requests | List of pull requests with their title and URL | List[PRItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Make Pull Request

### What it is
This block creates a new pull request on a specified GitHub repository.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| title | Title of the pull request | str | Yes |
| body | Body of the pull request | str | Yes |
| head | The name of the branch where your changes are implemented. For cross-repository pull requests in the same network, namespace head with a user like this: username:branch. | str | Yes |
| base | The name of the branch you want the changes pulled into. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the pull request creation failed | str |
| number | Number of the created pull request | int |
| url | URL of the created pull request | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Merge Pull Request

### What it is
This block merges a pull request using merge, squash, or rebase.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| merge_method | Merge method to use: merge, squash, or rebase | "merge" \| "squash" \| "rebase" | No |
| commit_title | Title for the merge commit (optional, used for merge and squash) | str | No |
| commit_message | Message for the merge commit (optional, used for merge and squash) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the merge failed | str |
| sha | SHA of the merge commit | str |
| merged | Whether the PR was merged | bool |
| message | Merge status message | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Read Pull Request

### What it is
This block reads the body, title, user, and changes of a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| include_pr_changes | Whether to include the changes made in the pull request | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reading the pull request failed | str |
| title | Title of the pull request | str |
| body | Body of the pull request | str |
| author | User who created the pull request | str |
| changes | Changes made in the pull request | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Unassign PR Reviewer

### What it is
This block unassigns a reviewer from a specified GitHub pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| pr_url | URL of the GitHub pull request | str | Yes |
| reviewer | Username of the reviewer to unassign | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the reviewer unassignment failed | str |
| status | Status of the reviewer unassignment operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
