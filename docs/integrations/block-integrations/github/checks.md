# GitHub Checks
<!-- MANUAL: file_description -->
Blocks for creating and updating GitHub check runs, enabling custom CI integration and detailed status reporting on commits and pull requests.
<!-- END MANUAL -->

## Github Create Check Run

### What it is
Creates a new check run for a specific commit in a GitHub repository

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new check run associated with a specific commit using the GitHub Checks API. Check runs represent individual test suites, linting tools, or other CI processes that report status against commits or pull requests.

You specify the commit SHA, check name, and current status. For completed checks, provide a conclusion (success, failure, or neutral) and optional detailed output including title, summary, and extended text for rich reporting in the GitHub UI.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| name | The name of the check run (e.g., 'code-coverage') | str | Yes |
| head_sha | The SHA of the commit to check | str | Yes |
| status | Current status of the check run | "queued" \| "in_progress" \| "completed" \| "waiting" \| "requested" \| "pending" | No |
| conclusion | The final conclusion of the check (required if status is completed) | "success" \| "failure" \| "neutral" \| "cancelled" \| "timed_out" \| "action_required" \| "skipped" | No |
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
**Custom CI Integration**: Create check runs for external CI systems that aren't natively integrated with GitHub.

**Code Quality Reporting**: Report linting, security scan, or test coverage results directly on commits and PRs.

**Deployment Status**: Track deployment progress by creating check runs that show pending, in-progress, and completed states.
<!-- END MANUAL -->

---

## Github Update Check Run

### What it is
Updates an existing check run in a GitHub repository

### How it works
<!-- MANUAL: how_it_works -->
This block updates an existing check run's status, conclusion, and output details via the GitHub Checks API. Use it to report progress as your CI process advances through different stages.

You can update the status from queued to in_progress to completed, and set the final conclusion when done. The output fields allow you to provide detailed results, annotations, and summaries visible in the GitHub UI.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo_url | URL of the GitHub repository | str | Yes |
| check_run_id | The ID of the check run to update | int | Yes |
| status | New status of the check run | "queued" \| "in_progress" \| "completed" \| "waiting" \| "requested" \| "pending" | Yes |
| conclusion | The final conclusion of the check (required if status is completed) | "success" \| "failure" \| "neutral" \| "cancelled" \| "timed_out" \| "action_required" \| "skipped" | Yes |
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
**Progress Reporting**: Update check runs as your CI pipeline progresses through build, test, and deployment stages.

**Real-Time Feedback**: Provide immediate feedback on pull requests as tests complete, rather than waiting for the entire suite.

**Failure Details**: Update check runs with detailed error messages and output when tests fail.
<!-- END MANUAL -->

---
