# Github Create Check Run

### What it is
Creates a new check run for a specific commit in a GitHub repository.

### What it does
Creates a new check run for a specific commit in a GitHub repository

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| name | The name of the check run (e.g., 'code-coverage') | str | Yes |
| head_sha | The SHA of the commit to check | str | Yes |
| status | Current status of the check run | "queued" | "in_progress" | "completed" | No |
| conclusion | The final conclusion of the check (required if status is completed) | "success" | "failure" | "neutral" | No |
| details_url | The URL for the full details of the check | str | No |
| output_title | Title of the check run output | str | No |
| output_summary | Summary of the check run output | str | No |
| output_text | Detailed text of the check run output | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if check run creation failed | str |
| check_run | Details of the created check run | CheckRunResult |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Github Update Check Run

### What it is
Updates an existing check run in a GitHub repository.

### What it does
Updates an existing check run in a GitHub repository

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| check_run_id | The ID of the check run to update | int | Yes |
| status | New status of the check run | "queued" | "in_progress" | "completed" | Yes |
| conclusion | The final conclusion of the check (required if status is completed) | "success" | "failure" | "neutral" | Yes |
| output_title | New title of the check run output | str | No |
| output_summary | New summary of the check run output | str | No |
| output_text | New detailed text of the check run output | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| check_run | Details of the updated check run | CheckRunResult |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
