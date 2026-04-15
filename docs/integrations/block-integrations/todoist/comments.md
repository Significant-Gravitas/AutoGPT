# Todoist Comments
<!-- MANUAL: file_description -->
Blocks for creating and managing comments on Todoist tasks and projects.
<!-- END MANUAL -->

## Todoist Create Comment

### What it is
Creates a new comment on a Todoist task or project

### How it works
<!-- MANUAL: how_it_works -->
Takes comment content and task/project ID, creates comment via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| content | Comment content | str | Yes |
| id_type | Specify either task_id or project_id to comment on | Id Type | No |
| attachment | Optional file attachment | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | ID of created comment | str |
| content | Comment content | str |
| posted_at | Comment timestamp | str |
| task_id | Associated task ID | str |
| project_id | Associated project ID | str |

### Possible use case
<!-- MANUAL: use_case -->
Adding notes and comments to tasks or projects automatically.
<!-- END MANUAL -->

---

## Todoist Delete Comment

### What it is
Deletes a Todoist comment

### How it works
<!-- MANUAL: how_it_works -->
Uses comment ID to delete via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | Comment ID to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the deletion was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
Removing outdated or incorrect comments from tasks/projects.
<!-- END MANUAL -->

---

## Todoist Get Comment

### What it is
Get a single comment from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Uses comment ID to retrieve details via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | Comment ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content | Comment content | str |
| id | Comment ID | str |
| posted_at | Comment timestamp | str |
| project_id | Associated project ID | str |
| task_id | Associated task ID | str |
| attachment | Optional file attachment | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
Looking up specific comment details for reference.
<!-- END MANUAL -->

---

## Todoist Get Comments

### What it is
Get all comments for a Todoist task or project

### How it works
<!-- MANUAL: how_it_works -->
Uses task/project ID to get comments list via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| id_type | Specify either task_id or project_id to get comments for | Id Type | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comments | List of comments | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
Reviewing comment history on tasks or projects.
<!-- END MANUAL -->

---

## Todoist Update Comment

### What it is
Updates a Todoist comment

### How it works
<!-- MANUAL: how_it_works -->
Takes comment ID and new content, updates via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | Comment ID to update | str | Yes |
| content | New content for the comment | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the update was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
Modifying existing comments to fix errors or update information.
<!-- END MANUAL -->

---
