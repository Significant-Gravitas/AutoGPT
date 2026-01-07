# Github Create Status

### What it is
Creates a new commit status in a GitHub repository.

### What it does
Creates a new commit status in a GitHub repository

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| sha | The SHA of the commit to set status for | str | Yes |
| state | The state of the status (error, failure, pending, success) | "error" | "failure" | "pending" | Yes |
| target_url | URL with additional details about this status | str | No |
| description | Short description of the status | str | No |
| check_name | Label to differentiate this status from others | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Details of the created status | StatusResult |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
