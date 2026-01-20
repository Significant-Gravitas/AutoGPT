# Todoist Sections
<!-- MANUAL: file_description -->
Blocks for managing sections within Todoist projects.
<!-- END MANUAL -->

## Todoist Delete Section

### What it is
Deletes a section and all its tasks from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Uses section ID to delete via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| section_id | ID of section to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether section was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
Removing unused sections or reorganizing projects.
<!-- END MANUAL -->

---

## Todoist Get Section

### What it is
Gets a single section by ID from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Uses section ID to retrieve details via Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| section_id | ID of section to fetch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | ID of section | str |
| project_id | Project ID the section belongs to | str |
| order | Order of the section | int |
| name | Name of the section | str |

### Possible use case
<!-- MANUAL: use_case -->
Looking up section details for task management.
<!-- END MANUAL -->

---

## Todoist List Sections

### What it is
Gets all sections and their details from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Connects to Todoist API to retrieve sections list.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | Optional project ID to filter sections | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| names_list | List of section names | List[str] |
| ids_list | List of section IDs | List[str] |
| complete_data | Complete section data including all fields | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
Getting section information for task organization.
<!-- END MANUAL -->

---
