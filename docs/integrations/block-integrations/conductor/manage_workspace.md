# Conductor Manage Workspace
<!-- MANUAL: file_description -->
Renames, archives, unarchives or sleeps a Conductor workspace, shares or stops its public preview URL, or moves it into a section. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Manage Workspace

### What it is
Change a Conductor workspace: rename it, archive, unarchive or sleep it, share or stop sharing a port at its public preview URL, or move it into a section.

### How it works
<!-- MANUAL: how_it_works -->
One `action` per run. `rename` posts `{name}` to `/rename`; `archive`, `unarchive` and `sleep` post to the matching lifecycle route and return the new `status`; `share_preview` puts `{port}` to `/preview` and returns the public `preview_url`; `stop_preview` deletes it; `move_to_section` puts `{sectionId}` to `/section`, with an empty `section_id` clearing the section. `result` always carries the raw response.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| workspace_id | Workspace ID | str | Yes |
| action | rename, archive, unarchive, sleep, share_preview (expose a port at a public preview URL), stop_preview, or move_to_section | "rename" \| "archive" \| "unarchive" \| "sleep" \| "share_preview" \| "stop_preview" \| "move_to_section" | No |
| name | New workspace name (rename) | str | No |
| port | Port inside the workspace to share (share_preview) | int | No |
| section_id | Target section ID (move_to_section); empty clears the section | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| workspace_id | Workspace ID | str |
| status | Workspace state after the action, when reported | str |
| preview_url | Preview URL after a share_preview action, else empty | str |
| result | Raw API response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Share a running app**: After an agent starts a dev server, share port 3000 and hand the `preview_url` to a reviewer.

**Tidy up**: Archive workspaces whose pull request merged, or sleep idle ones to free resources.

**File by project**: Move new workspaces into a per-customer section created with Manage Section.
<!-- END MANUAL -->

---
