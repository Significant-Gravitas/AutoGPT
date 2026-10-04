# Capy Projects
<!-- MANUAL: file_description -->
Find the Capy projects an API key can reach. A project groups the repositories a Capy agent works in, and every new thread needs a project ID.
<!-- END MANUAL -->

## Capy List Projects

### What it is
Lists the Capy projects your API key can see, with the repositories each one covers. Use it to find the project ID a new Capy thread runs in.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/projects` with the API key and returns each project with its repositories and base branches. A key sees every project its principal can access; a service-user key sees only its allowlisted projects.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| projects | Every project visible to the key, with its repositories | List[Project] |
| project | Each project, one at a time | Project |
| project_ids | IDs of the projects, for the project_id input elsewhere | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Project Lookup**: Resolve the project for "the checkout repo" before starting a thread.

**Repository Routing**: Let an agent pick the project whose repositories match the work it was handed.

**Access Audit**: List which projects and repositories an integration's key can reach.
<!-- END MANUAL -->

---
