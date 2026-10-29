# Conductor Account
<!-- MANUAL: file_description -->
One-call overview of a Conductor account: identity, projects (repositories), sections and routines. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Get Account

### What it is
Get an overview of your Conductor account in one call: who you are, the projects (repositories) you can open workspaces in, your sections and your routines. Use this first to find project IDs.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /me` for the caller's identity, then pages through `GET /v0/projects`, `GET /v0/sections` and `GET /v0/routines` (each `{data, offset, hasMore}` listing) until `limit` items or the end. Routine listings never include webhook URLs; use Manage Routine to create a routine or rotate its secret and receive the URL. Any HTTP or API error (`userMessage` from Conductor) is raised as a block error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Maximum number of projects, sections and routines to list | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| user | The authenticated identity: userId, name, email, organizationId | Dict[str, Any] |
| projects | Repositories you can create workspaces in: id, name, gitRemote | List[Dict[str, Any]] |
| sections | Your cloud sections: id, name, emoji, workspaceIds | List[Dict[str, Any]] |
| routines | Your routines: id, name, prompt, repoUrl, agent, model, enabled, runCount, triggers (webhook URLs are not included) | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Orient an agent**: Run this first so AutoPilot or a graph can look up the `projects[].id` it needs before creating a workspace.

**Inventory**: List sections and routines to decide where new workspaces should be filed and which automations already exist.
<!-- END MANUAL -->

---
