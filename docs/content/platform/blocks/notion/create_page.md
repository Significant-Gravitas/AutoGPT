# Notion Create Page

### What it is
Create a new page in Notion.

### What it does
Create a new page in Notion. Requires EITHER a parent_page_id OR parent_database_id. Supports markdown content.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| parent_page_id | Parent page ID to create the page under. Either this OR parent_database_id is required. | str | No |
| parent_database_id | Parent database ID to create the page in. Either this OR parent_page_id is required. | str | No |
| title | Title of the new page | str | Yes |
| content | Content for the page. Can be plain text or markdown - will be converted to Notion blocks. | str | No |
| properties | Additional properties for database pages (e.g., {'Status': 'In Progress', 'Priority': 'High'}) | Dict[str, True] | No |
| icon_emoji | Emoji to use as the page icon (e.g., '📄', '🚀') | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| page_id | ID of the created page. | str |
| page_url | URL of the created page. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
