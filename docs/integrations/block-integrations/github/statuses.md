# GitHub Statuses
<!-- MANUAL: file_description -->
Blocks for creating and managing GitHub commit statuses for CI/CD integration.
<!-- END MANUAL -->

## Github Create Status

### What it is
Creates a new commit status in a GitHub repository

### How it works
<!-- MANUAL: how_it_works -->
This block creates a commit status using the GitHub Status API. Commit statuses are simpler than check runs and appear as colored indicators (pending yellow, success green, failure red, error red) on commits and pull requests.

Provide a context label to differentiate this status from others, an optional target URL for detailed results, and a description. Multiple statuses can exist on the same commit with different context labels.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| sha | The SHA of the commit to set status for | str | Yes |
| state | The state of the status (error, failure, pending, success) | "error" \| "failure" \| "pending" \| "success" | Yes |
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
**External CI Integration**: Report build status from CI systems that don't have native GitHub integration.

**Deployment Tracking**: Set commit statuses to indicate deployment state (pending, deployed, failed).

**Required Status Checks**: Create statuses that GitHub branch protection rules require before merging.
<!-- END MANUAL -->

---
