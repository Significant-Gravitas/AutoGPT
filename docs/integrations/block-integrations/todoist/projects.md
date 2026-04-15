# Todoist Projects
<!-- MANUAL: file_description -->
Blocks for creating and managing projects in Todoist.
<!-- END MANUAL -->

## Todoist Create Project

### What it is
Creates a new project in Todoist

### How it works
<!-- MANUAL: how_it_works -->
Takes project details and creates via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | Name of the project | str | Yes |
| parent_id | Parent project ID | str | No |
| color | Color of the project icon | "berry_red" \| "red" \| "orange" \| "yellow" \| "olive_green" \| "lime_green" \| "green" \| "mint_green" \| "teal" \| "sky_blue" \| "light_blue" \| "blue" \| "grape" \| "violet" \| "lavender" \| "magenta" \| "salmon" \| "charcoal" \| "grey" \| "taupe" | No |
| is_favorite | Whether the project is a favorite | bool | No |
| view_style | Display style (list or board) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the creation was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
Creating new projects programmatically for workflow automation.
<!-- END MANUAL -->

---

## Todoist Delete Project

### What it is
Deletes a Todoist project and all its contents

### How it works
<!-- MANUAL: how_it_works -->
Uses project ID to delete via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | ID of project to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the deletion was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
Removing completed or obsolete projects.
<!-- END MANUAL -->

---

## Todoist Get Project

### What it is
Gets details for a specific Todoist project

### How it works
<!-- MANUAL: how_it_works -->
Uses project ID to retrieve details via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | ID of the project to get details for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| project_id | ID of project | str |
| project_name | Name of project | str |
| project_url | URL of project | str |
| complete_data | Complete project data including all fields | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
Looking up project details for verification or editing.
<!-- END MANUAL -->

---

## Todoist List Collaborators

### What it is
Gets all collaborators for a specific Todoist project

### How it works
<!-- MANUAL: how_it_works -->
Uses project ID to get collaborator list via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | ID of the project to get collaborators for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| collaborator_ids | List of collaborator IDs | List[str] |
| collaborator_names | List of collaborator names | List[str] |
| collaborator_emails | List of collaborator email addresses | List[str] |
| complete_data | Complete collaborator data including all fields | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
Managing project sharing and collaboration.
<!-- END MANUAL -->

---

## Todoist List Projects

### What it is
Gets all projects and their details from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Connects to Todoist API using provided credentials and retrieves all projects.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| names_list | List of project names | List[str] |
| ids_list | List of project IDs | List[str] |
| url_list | List of project URLs | List[str] |
| complete_data | Complete project data including all fields | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
Getting an overview of all projects for organization or automation.
<!-- END MANUAL -->

---

## Todoist Update Project

### What it is
Updates an existing project in Todoist

### How it works
<!-- MANUAL: how_it_works -->
Takes project ID and updated fields, applies via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | ID of project to update | str | Yes |
| name | New name for the project | str | No |
| color | New color for the project icon | "berry_red" \| "red" \| "orange" \| "yellow" \| "olive_green" \| "lime_green" \| "green" \| "mint_green" \| "teal" \| "sky_blue" \| "light_blue" \| "blue" \| "grape" \| "violet" \| "lavender" \| "magenta" \| "salmon" \| "charcoal" \| "grey" \| "taupe" | No |
| is_favorite | Whether the project should be a favorite | bool | No |
| view_style | Display style (list or board) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the update was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
Modifying project settings or reorganizing projects.
<!-- END MANUAL -->

---
