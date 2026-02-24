# Linear Projects
<!-- MANUAL: file_description -->
Blocks for searching and managing Linear projects.
<!-- END MANUAL -->

## Linear Search Projects

### What it is
Searches for projects on Linear

### How it works
<!-- MANUAL: how_it_works -->
This block searches for projects in Linear using a text query. It queries the Linear GraphQL API to find projects matching the search term.

Returns a list of projects with their details for further use in workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| term | Term to search for projects | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| projects | List of projects | List[Project] |

### Possible use case
<!-- MANUAL: use_case -->
**Project Discovery**: Find projects by name to use in issue creation or queries.

**Portfolio Overview**: Search for projects to build portfolio dashboards.

**Dynamic Forms**: Populate project dropdowns in custom interfaces.
<!-- END MANUAL -->

---
