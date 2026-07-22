# GitHub Issues
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Github Add Label

### What it is
A block that adds a label to a GitHub issue or pull request for categorization and organization.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue or pull request | str | Yes |
| label | Label to add to the issue or pull request | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the label addition failed | str |
| status | Status of the label addition operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Assign Issue

### What it is
A block that assigns a GitHub user to an issue for task ownership and tracking.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue | str | Yes |
| assignee | Username to assign to the issue | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the issue assignment failed | str |
| status | Status of the issue assignment operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Comment

### What it is
A block that posts comments on GitHub issues or pull requests using the GitHub API.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue or pull request | str | Yes |
| comment | Comment to post on the issue or pull request | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the comment posting failed | str |
| id | ID of the created comment | int |
| url | URL to the comment on GitHub | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Comments

### What it is
A block that retrieves all comments from a GitHub issue or pull request, including comment metadata and content.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue or pull request | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comment | Comments with their ID, body, user, and URL | Comment |
| comments | List of comments with their ID, body, user, and URL | List[CommentItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github List Issues

### What it is
A block that retrieves a list of issues from a GitHub repository with their titles and URLs.

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
| issue | Issues with their title and URL | Issue |
| issues | List of issues with their title and URL | List[IssueItem] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Make Issue

### What it is
A block that creates new issues on GitHub repositories with a title and body content.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| title | Title of the issue | str | Yes |
| body | Body of the issue | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the issue creation failed | str |
| number | Number of the created issue | int |
| url | URL of the created issue | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Read Issue

### What it is
A block that retrieves information about a specific GitHub issue, including its title, body content, and creator.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reading the issue failed | str |
| title | Title of the issue | str |
| body | Body of the issue | str |
| user | User who created the issue | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Remove Label

### What it is
A block that removes a label from a GitHub issue or pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue or pull request | str | Yes |
| label | Label to remove from the issue or pull request | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the label removal failed | str |
| status | Status of the label removal operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Unassign Issue

### What it is
A block that removes a user's assignment from a GitHub issue.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| issue_url | URL of the GitHub issue | str | Yes |
| assignee | Username to unassign from the issue | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the issue unassignment failed | str |
| status | Status of the issue unassignment operation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Update Comment

### What it is
A block that updates an existing comment on a GitHub issue or pull request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_url | URL of the GitHub comment | str | No |
| issue_url | URL of the GitHub issue or pull request | str | No |
| comment_id | ID of the GitHub comment | str | No |
| comment | Comment to update | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the comment update failed | str |
| id | ID of the updated comment | int |
| url | URL to the comment on GitHub | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
