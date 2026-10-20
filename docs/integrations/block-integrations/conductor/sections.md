# Conductor Sections
<!-- MANUAL: file_description -->
Creates or deletes a Conductor cloud section, the sidebar groups that organise workspaces. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Manage Section

### What it is
Create or delete a Conductor cloud section. Sections group workspaces in the sidebar; move workspaces with Manage Workspace.

### How it works
<!-- MANUAL: how_it_works -->
`create` posts `{name, emoji}` to `POST /v0/sections` and returns the new section; `delete` calls `DELETE /v0/sections/{id}` and returns the section together with the workspace ids that were in it (`removed_workspace_ids`), which are left unfiled rather than deleted. Move workspaces between sections with Manage Workspace's `move_to_section` action.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| action | create a personal section, or delete one | "create" \| "delete" | No |
| name | Section name (create) | str | No |
| emoji | Optional emoji shown next to the section name (create) | str | No |
| section_id | Section ID (delete) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| section_id | ID of the created or deleted section | str |
| section | The section: id, name, emoji, workspaceIds | Dict[str, Any] |
| removed_workspace_ids | Workspaces that were in the section (delete only) | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Per-customer folders**: Create a section per client and move their workspaces into it.

**Retire a project**: Delete its section and archive the returned workspace ids.
<!-- END MANUAL -->

---
