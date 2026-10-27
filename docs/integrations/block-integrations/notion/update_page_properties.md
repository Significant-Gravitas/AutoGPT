# Notion Update Page Properties
<!-- MANUAL: file_description -->
Blocks for changing pages that already exist in Notion workspaces and databases.
<!-- END MANUAL -->

## Notion Update Page Properties

### What it is
Update an existing Notion page's properties, its title among them, its icon or cover, or whether it is in the trash. The page's content is left as it is.

### How it works
<!-- MANUAL: how_it_works -->
This block updates an existing page through the Notion API. It can rename the page, set its properties, change its emoji icon or cover image, and move it to the trash or restore it, all in one request. The page must be shared with your integration. It leaves the page's content as it is; the Notion Append To Page block adds content to the end of a page.

Properties are named as they appear in the database, and each value is written as the type that property already has: a name for a select or status, a list of names for a multi-select, a date string for a date, and so on. The block reads the page first to learn those types. Before anything is written, it refuses a property the page does not have, a read-only one such as a formula, and a value for a people, relation or files property. A null value clears the property and unchecks a checkbox, but a status goes back to its default option if it has one.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| page_id | ID of the page to update. Must be accessible by the connected integration. | str | Yes |
| title | New title for the page | str | No |
| properties | Properties to set by name (e.g., {'Status': 'Done', 'Tags': ['urgent']}). Each value is written as the type the property already has; null clears it, or resets a status to its default option. | Dict[str, Any] | No |
| icon_emoji | Emoji to use as the page icon (e.g., '📄', '🚀') | str | No |
| cover_url | URL of an image to use as the page cover | str | No |
| in_trash | True moves the page to the trash, False restores it. Leave empty to keep it where it is. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| page_id | ID of the updated page. | str |
| page_url | URL of the updated page. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Task Status Sync**: Mark a task page as Done in a Notion database when the matching issue closes in another system.

**Content Workflow**: Move a draft through Review and Published by setting its Status property as each step finishes.

**Cleanup**: Move pages that a workflow no longer needs to the trash, where they can still be restored.
<!-- END MANUAL -->

---
