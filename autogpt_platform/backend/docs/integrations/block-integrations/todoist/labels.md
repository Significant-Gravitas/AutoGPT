# Todoist Labels
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Todoist Create Label

### What it is
Creates a new label in Todoist, It will not work if same name already exists

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist Delete Label

### What it is
Deletes a personal label in Todoist

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist Get Label

### What it is
Gets a personal label from Todoist by ID

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist Get Shared Labels

### What it is
Gets all shared labels from Todoist

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| labels | List of shared label names | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist List Labels

### What it is
Gets all personal labels from Todoist

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist Remove Shared Labels

### What it is
Removes all instances of a shared label

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist Rename Shared Labels

### What it is
Renames all instances of a shared label

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Todoist Update Label

### What it is
Updates a personal label in Todoist

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
