# GitHub Issues
<!-- MANUAL: file_description -->
Blocks for managing GitHub issues including creating, reading, listing, commenting, labeling, and assigning issues programmatically.
<!-- END MANUAL -->

## Github Add Label

### What it is
A block that adds a label to a GitHub issue or pull request for categorization and organization.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials, the URL of the issue or pull request, and the label to be added as inputs. It then sends a request to the GitHub API to add the label to the specified issue or pull request.
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
Automatically categorizing issues based on their content or assigning priority labels to newly created issues.
<!-- END MANUAL -->

---

## Github Assign Issue

### What it is
A block that assigns a GitHub user to an issue for task ownership and tracking.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials, the URL of the issue, and the username of the person to be assigned as inputs. It then sends a request to the GitHub API to assign the specified user to the issue.
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
Automatically assigning new issues to team members based on their expertise or workload.
<!-- END MANUAL -->

---

## Github Comment

### What it is
A block that posts comments on GitHub issues or pull requests using the GitHub API.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials, the URL of the issue or pull request, and the comment text as inputs. It then sends a request to the GitHub API to post the comment on the specified issue or pull request.
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
Automating responses to issues in a GitHub repository, such as thanking contributors for their submissions or providing status updates on reported bugs.
<!-- END MANUAL -->

---

## Github List Comments

### What it is
A block that retrieves all comments from a GitHub issue or pull request, including comment metadata and content.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all comments from a GitHub issue or pull request via the GitHub API. It authenticates using your GitHub credentials and fetches the complete comment history, returning both individual comments and a list of all comments with their metadata.

Each comment includes the comment ID, body text, author username, and a direct URL to the comment on GitHub.
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
**Conversation Analysis**: Extract all comments from an issue to analyze the discussion or generate a summary of the conversation.

**Comment Monitoring**: Track all responses on specific issues to monitor team communication or customer feedback.

**Audit Trails**: Collect comment history for compliance or documentation purposes.
<!-- END MANUAL -->

---

## Github List Issues

### What it is
A block that retrieves a list of issues from a GitHub repository with their titles and URLs.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials and repository URL as inputs. It then sends a request to the GitHub API to fetch the list of issues and returns their details.
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
Creating a summary of open issues for a project status report or displaying them on a project management dashboard.
<!-- END MANUAL -->

---

## Github Make Issue

### What it is
A block that creates new issues on GitHub repositories with a title and body content.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials, repository URL, issue title, and issue body as inputs. It then sends a request to the GitHub API to create a new issue with the provided information.
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
Automatically creating issues for bug reports or feature requests submitted through an external system or form.
<!-- END MANUAL -->

---

## Github Read Issue

### What it is
A block that retrieves information about a specific GitHub issue, including its title, body content, and creator.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials and the issue URL as inputs. It then sends a request to the GitHub API to fetch the issue's details and returns the relevant information.
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
Gathering information about reported issues for analysis or to display on a dashboard.
<!-- END MANUAL -->

---

## Github Remove Label

### What it is
A block that removes a label from a GitHub issue or pull request.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials, the URL of the issue or pull request, and the label to be removed as inputs. It then sends a request to the GitHub API to remove the label from the specified issue or pull request.
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
Updating the status of issues as they progress through a workflow, such as removing a "In Progress" label when an issue is completed.
<!-- END MANUAL -->

---

## Github Unassign Issue

### What it is
A block that removes a user's assignment from a GitHub issue.

### How it works
<!-- MANUAL: how_it_works -->
The block takes the GitHub credentials, the URL of the issue, and the username of the person to be unassigned as inputs. It then sends a request to the GitHub API to remove the specified user's assignment from the issue.
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
Automatically unassigning issues that have been inactive for a certain period or when reassigning workload among team members.
<!-- END MANUAL -->

---

## Github Update Comment

### What it is
A block that updates an existing comment on a GitHub issue or pull request.

### How it works
<!-- MANUAL: how_it_works -->
This block updates an existing comment on a GitHub issue or pull request. You can identify the comment to update using either the direct comment URL, or a combination of the issue URL and comment ID. The block sends a PATCH request to the GitHub API to replace the comment's content.

The updated comment retains its original author and timestamp context while replacing the body text with your new content.
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
**Status Updates**: Modify a pinned status comment to reflect current progress on an issue.

**Bot Maintenance**: Update automated bot comments with new information instead of creating duplicate comments.

**Error Corrections**: Fix typos or incorrect information in previously posted comments.
<!-- END MANUAL -->

---
