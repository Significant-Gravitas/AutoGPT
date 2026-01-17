# Linear Issues
<!-- MANUAL: file_description -->
Blocks for creating and managing Linear issues.
<!-- END MANUAL -->

## Linear Create Issue

### What it is
Creates a new issue on Linear

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new issue in Linear using the GraphQL API. Specify the team, title, description, and optionally priority and project. The issue is created immediately and assigned to the specified team's workflow.

Returns the created issue's ID and title for tracking or further operations.
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
**Bug Reporting**: Automatically create issues from error monitoring or customer reports.

**Feature Requests**: Convert feature requests from forms or support tickets into Linear issues.

**Task Automation**: Create issues based on scheduled events or external triggers.
<!-- END MANUAL -->

---

## Linear Get Project Issues

### What it is
Gets issues from a Linear project filtered by status and assignee

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves issues from a Linear project with optional filtering by status and assignee. It queries the Linear GraphQL API and returns matching issues with their details.

Optionally include comments in the response for comprehensive issue data.
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
**Sprint Reports**: Generate reports of issues in specific states for sprint reviews.

**Workload Analysis**: Find unassigned or overdue issues across projects.

**Status Dashboards**: Build dashboards showing issue distribution by status.
<!-- END MANUAL -->

---

## Linear Search Issues

### What it is
Searches for issues on Linear

### How it works
<!-- MANUAL: how_it_works -->
This block searches for issues in Linear using a text query. It searches across issue titles, descriptions, and other fields to find matching issues.

Returns a list of issues matching the search term.
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
**Duplicate Detection**: Search for existing issues before creating new ones.

**Related Issues**: Find issues related to a specific topic or feature.

**Quick Lookup**: Search for issues by keyword for customer support or research.
<!-- END MANUAL -->

---
