# Linear Create Issue

### What it is
Creates a new issue on Linear.

### What it does
Creates a new issue on Linear

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Title of the issue | str | Yes |
| description | Description of the issue | str | Yes |
| team_name | Name of the team to create the issue on | str | Yes |
| priority | Priority of the issue | int | No |
| project_name | Name of the project to create the issue on | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| issue_id | ID of the created issue | str |
| issue_title | Title of the created issue | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Linear Get Project Issues

### What it is
Gets issues from a Linear project filtered by status and assignee.

### What it does
Gets issues from a Linear project filtered by status and assignee

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project | Name of the project to get issues from | str | Yes |
| status | Status/state name to filter issues by (e.g., 'In Progress', 'Done') | str | Yes |
| is_assigned | Filter by assignee status - True to get assigned issues, False to get unassigned issues | bool | No |
| include_comments | Whether to include comments in the response | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| issues | List of issues matching the criteria | List[Issue] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Linear Search Issues

### What it is
Searches for issues on Linear.

### What it does
Searches for issues on Linear

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| term | Term to search for issues | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| issues | List of issues | List[Issue] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
