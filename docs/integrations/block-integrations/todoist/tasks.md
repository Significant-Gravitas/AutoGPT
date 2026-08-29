# Todoist Tasks
<!-- MANUAL: file_description -->
Blocks for creating, updating, and managing tasks in Todoist.
<!-- END MANUAL -->

## Todoist Close Task

### What it is
Closes a task in Todoist

### How it works
<!-- MANUAL: how_it_works -->
Uses task ID to mark it complete via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task_id | Task ID to close | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the task was successfully closed | bool |

### Possible use case
<!-- MANUAL: use_case -->
Marking tasks as done in automated workflows.
<!-- END MANUAL -->

---

## Todoist Create Task

### What it is
Creates a new task in a Todoist project

### How it works
<!-- MANUAL: how_it_works -->
Takes task details and creates a new task via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| content | Task content | str | Yes |
| description | Task description | str | No |
| project_id | Project ID this task should belong to | str | No |
| section_id | Section ID this task should belong to | str | No |
| parent_id | Parent task ID | str | No |
| order | Optional order among other tasks,[Non-zero integer value used by clients to sort tasks under the same parent] | int | No |
| labels | Task labels | List[str] | No |
| priority | Task priority from 1 (normal) to 4 (urgent) | int | No |
| due_date | Due date in YYYY-MM-DD format | str (date-time) | No |
| deadline_date | Specific date in YYYY-MM-DD format relative to user's timezone | str (date-time) | No |
| assignee_id | Responsible user ID | str | No |
| duration_unit | Task duration unit (minute/day) | str | No |
| duration | Task duration amount, You need to selecct the duration unit first | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | Task ID | str |
| url | Task URL | str |
| complete_data | Complete task data as dictionary | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
Creating new tasks with full customization of parameters.
<!-- END MANUAL -->

---

## Todoist Delete Task

### What it is
Deletes a task in Todoist

### How it works
<!-- MANUAL: how_it_works -->
Uses task ID to delete via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task_id | Task ID to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the task was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
Removing unwanted or obsolete tasks from the system.
<!-- END MANUAL -->

---

## Todoist Get Task

### What it is
Get an active task from Todoist

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a single active task from Todoist using the task's unique ID. It queries the Todoist REST API and returns comprehensive task details including content, description, due dates, labels, and project association.

Only active (uncompleted) tasks can be retrieved; closed tasks are not accessible through this endpoint.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task_id | Task ID to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| project_id | Project ID containing the task | str |
| url | Task URL | str |
| complete_data | Complete task data as dictionary | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Task Details Retrieval**: Fetch complete information about a specific task for display or processing in workflows.

**Workflow Branching**: Get task details to make decisions based on due dates, labels, or priority levels.

**Task Auditing**: Retrieve individual tasks to verify their current state before performing updates or other operations.
<!-- END MANUAL -->

---

## Todoist Get Tasks

### What it is
Get active tasks from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Queries Todoist API with provided filters to get matching tasks.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | Filter tasks by project ID | str | No |
| section_id | Filter tasks by section ID | str | No |
| label | Filter tasks by label name | str | No |
| filter | Filter by any supported filter, You can see How to use filters or create one of your one here - https://todoist.com/help/articles/introduction-to-filters-V98wIH | str | No |
| lang | IETF language tag for filter language | str | No |
| ids | List of task IDs to retrieve | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | Task IDs | List[str] |
| urls | Task URLs | List[str] |
| complete_data | Complete task data as dictionary | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
Retrieving tasks matching specific criteria for review or processing.
<!-- END MANUAL -->

---

## Todoist Reopen Task

### What it is
Reopens a task in Todoist

### How it works
<!-- MANUAL: how_it_works -->
Uses task ID to reactivate via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task_id | Task ID to reopen | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the task was successfully reopened | bool |

### Possible use case
<!-- MANUAL: use_case -->
Reactivating tasks that were closed accidentally or need to be repeated.
<!-- END MANUAL -->

---

## Todoist Update Task

### What it is
Updates an existing task in Todoist

### How it works
<!-- MANUAL: how_it_works -->
Takes task ID and updated fields, applies changes via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task_id | Task ID to update | str | Yes |
| content | Task content | str | Yes |
| description | Task description | str | No |
| project_id | Project ID this task should belong to | str | No |
| section_id | Section ID this task should belong to | str | No |
| parent_id | Parent task ID | str | No |
| order | Optional order among other tasks,[Non-zero integer value used by clients to sort tasks under the same parent] | int | No |
| labels | Task labels | List[str] | No |
| priority | Task priority from 1 (normal) to 4 (urgent) | int | No |
| due_date | Due date in YYYY-MM-DD format | str (date-time) | No |
| deadline_date | Specific date in YYYY-MM-DD format relative to user's timezone | str (date-time) | No |
| assignee_id | Responsible user ID | str | No |
| duration_unit | Task duration unit (minute/day) | str | No |
| duration | Task duration amount, You need to selecct the duration unit first | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the update was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
Modifying task details like due dates, priority etc.
<!-- END MANUAL -->

---
