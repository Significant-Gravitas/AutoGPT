# GitHub CI
<!-- MANUAL: file_description -->
Blocks for retrieving CI check results from commits or pull requests, including overall status, pass/fail counts, and optional log searching.
<!-- END MANUAL -->

## Github Get CI Results

### What it is
This block gets CI results for a commit or PR, with optional search for specific errors/warnings in logs.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves CI check results for a specific commit or pull request using the GitHub Checks API. It aggregates results from all CI checks, providing an overall status summary along with individual check details.

Optionally search through CI logs using regex patterns to find specific errors or warnings. You can filter by check name to focus on particular CI jobs. The block returns comprehensive results including pass/fail counts and matched log lines.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| repo | GitHub repository | str | Yes |
| target | Commit SHA or PR number to get CI results for | str \| int | Yes |
| search_pattern | Optional regex pattern to search for in CI logs (e.g., error messages, file names) | str | No |
| check_name_filter | Optional filter for specific check names (supports wildcards) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| check_run | Individual CI check run with details | Check Run |
| check_runs | List of all CI check runs | List[CheckRunItem] |
| matched_line | Line matching the search pattern with context | Matched Line |
| matched_lines | All lines matching the search pattern across all checks | List[MatchedLine] |
| overall_status | Overall CI status (pending, success, failure) | str |
| overall_conclusion | Overall CI conclusion if completed | str |
| total_checks | Total number of CI checks | int |
| passed_checks | Number of passed checks | int |
| failed_checks | Number of failed checks | int |

### Possible use case
<!-- MANUAL: use_case -->
**CI Status Monitoring**: Check the overall CI status of commits or PRs before merging or deploying.

**Error Diagnosis**: Search CI logs for specific error patterns to quickly identify why builds are failing.

**Automated PR Validation**: Verify all required checks pass before automatically proceeding with merge or deployment workflows.
<!-- END MANUAL -->

---
