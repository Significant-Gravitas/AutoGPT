# Todoist Labels
<!-- MANUAL: file_description -->
Blocks for creating and managing labels in Todoist.
<!-- END MANUAL -->

## Todoist Create Label

### What it is
Creates a new label in Todoist, It will not work if same name already exists

### How it works
<!-- MANUAL: how_it_works -->
It takes label details as input, connects to Todoist API, creates the label and returns the created label's details.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | Name of the label | str | Yes |
| order | Label order | int | No |
| color | The color of the label icon | "berry_red" \| "red" \| "orange" \| "yellow" \| "olive_green" \| "lime_green" \| "green" \| "mint_green" \| "teal" \| "sky_blue" \| "light_blue" \| "blue" \| "grape" \| "violet" \| "lavender" \| "magenta" \| "salmon" \| "charcoal" \| "grey" \| "taupe" | No |
| is_favorite | Whether the label is a favorite | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | ID of the created label | str |
| name | Name of the label | str |
| color | Color of the label | str |
| order | Label order | int |
| is_favorite | Favorite status | bool |

### Possible use case
<!-- MANUAL: use_case -->
Creating new labels to organize and categorize tasks in Todoist.
<!-- END MANUAL -->

---

## Todoist Delete Label

### What it is
Deletes a personal label in Todoist

### How it works
<!-- MANUAL: how_it_works -->
This block permanently removes a personal label from Todoist using the label's unique ID. The deletion is processed through the Todoist REST API and removes the label from all tasks it was assigned to.

The operation is irreversible, so any tasks previously tagged with this label will lose that categorization after deletion.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| label_id | ID of the label to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the deletion was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Label Cleanup**: Remove obsolete labels when reorganizing your task management system.

**Workflow Automation**: Delete temporary labels after a project phase is complete.

**Bulk Management**: Remove labels as part of a larger cleanup workflow.
<!-- END MANUAL -->

---

## Todoist Get Label

### What it is
Gets a personal label from Todoist by ID

### How it works
<!-- MANUAL: how_it_works -->
Uses the label ID to retrieve label details from Todoist API.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| label_id | ID of the label to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | ID of the label | str |
| name | Name of the label | str |
| color | Color of the label | str |
| order | Label order | int |
| is_favorite | Favorite status | bool |

### Possible use case
<!-- MANUAL: use_case -->
Looking up details of a specific label for editing or verification.
<!-- END MANUAL -->

---

## Todoist Get Shared Labels

### What it is
Gets all shared labels from Todoist

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all shared labels that exist across collaborative projects in your Todoist account. Shared labels are labels that appear on tasks in projects shared with other users.

The API returns a list of label names that are currently in use across shared projects, enabling cross-project label management.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| labels | List of shared label names | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Collaboration Audit**: Review which labels are being used across shared projects.

**Label Consistency**: Ensure consistent labeling conventions across team projects.

**Cross-Project Analytics**: Analyze label usage patterns in collaborative workspaces.
<!-- END MANUAL -->

---

## Todoist List Labels

### What it is
Gets all personal labels from Todoist

### How it works
<!-- MANUAL: how_it_works -->
Connects to Todoist API using provided credentials and retrieves all labels.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| labels | List of complete label data | List[Any] |
| label_ids | List of label IDs | List[Any] |
| label_names | List of label names | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
Getting an overview of all labels to organize tasks or find specific labels.
<!-- END MANUAL -->

---

## Todoist Remove Shared Labels

### What it is
Removes all instances of a shared label

### How it works
<!-- MANUAL: how_it_works -->
This block removes a shared label by name from all tasks across all shared projects. Unlike deleting a personal label, this operation targets labels by name rather than ID since shared labels are name-based.

The removal affects all instances of the label across collaborative projects, untagging every task that had this label applied.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the label to remove | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the removal was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Deprecate Labels**: Remove labels that are no longer part of your workflow conventions.

**Team Cleanup**: Remove shared labels when reorganizing cross-project categorization.

**Merge Labels**: Remove a duplicate label after migrating tasks to a standardized label.
<!-- END MANUAL -->

---

## Todoist Rename Shared Labels

### What it is
Renames all instances of a shared label

### How it works
<!-- MANUAL: how_it_works -->
This block renames a shared label across all tasks in all shared projects. It takes the existing label name and a new name, then updates every instance where that label appears.

The rename is atomic across the entire account, ensuring consistent label naming in collaborative environments.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the existing label to rename | str | Yes |
| new_name | The new name for the label | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the rename was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Standardize Naming**: Rename labels to follow consistent naming conventions.

**Rebrand Categories**: Update label names when workflow terminology changes.

**Fix Typos**: Correct misspelled labels across all shared projects.
<!-- END MANUAL -->

---

## Todoist Update Label

### What it is
Updates a personal label in Todoist

### How it works
<!-- MANUAL: how_it_works -->
This block modifies an existing personal label's properties using the Todoist API. You can update the label's name, display order, color, and favorite status.

Only the fields you provide are updated; omitted fields retain their current values. The label ID is required to identify which label to modify.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| label_id | ID of the label to update | str | Yes |
| name | New name of the label | str | No |
| order | Label order | int | No |
| color | The color of the label icon | "berry_red" \| "red" \| "orange" \| "yellow" \| "olive_green" \| "lime_green" \| "green" \| "mint_green" \| "teal" \| "sky_blue" \| "light_blue" \| "blue" \| "grape" \| "violet" \| "lavender" \| "magenta" \| "salmon" \| "charcoal" \| "grey" \| "taupe" | No |
| is_favorite | Whether the label is a favorite (true/false) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the update was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Visual Organization**: Change label colors to create visual groupings of related labels.

**Priority Adjustment**: Update favorite status to surface frequently used labels.

**Reorganization**: Modify label order to reflect current workflow priorities.
<!-- END MANUAL -->

---
