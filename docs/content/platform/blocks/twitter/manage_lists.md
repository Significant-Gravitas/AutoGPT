# Twitter Create List

### What it is
This block creates a new Twitter List for the authenticated user.

### What it does
This block creates a new Twitter List for the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the List to be created | str | No |
| description | Description of the List | str | No |
| private | Whether the List should be private | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the created list | str |
| list_id | ID of the created list | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Delete List

### What it is
This block deletes a specified Twitter List owned by the authenticated user.

### What it does
This block deletes a specified Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to be deleted | str | Yes |

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

## Twitter Update List

### What it is
This block updates a specified Twitter List owned by the authenticated user.

### What it does
This block updates a specified Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to be updated | str | Yes |
| name | New name for the List | str | No |
| description | New description for the List | str | No |

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
