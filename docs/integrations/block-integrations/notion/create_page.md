# Notion Create Page
<!-- MANUAL: file_description -->
Blocks for creating new pages in Notion workspaces and databases.
<!-- END MANUAL -->

## Notion Create Page

### What it is
Create a new page in Notion. Requires EITHER a parent_page_id OR parent_database_id. Supports markdown content.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new page in Notion using the Notion API. You can create pages as children of existing pages or as entries in a database. The parent must be accessible to your integration.

Content can be provided as markdown, which gets converted to Notion blocks. For database pages, you can set additional properties like Status or Priority. Optionally add an emoji icon to the page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| parent_page_id | Parent page ID to create the page under. Either this OR parent_database_id is required. | str | No |
| parent_database_id | Parent database ID to create the page in. Either this OR parent_page_id is required. | str | No |
| title | Title of the new page | str | Yes |
| content | Content for the page. Can be plain text or markdown - will be converted to Notion blocks. | str | No |
| properties | Additional properties for database pages (e.g., {'Status': 'In Progress', 'Priority': 'High'}) | Dict[str, Any] | No |
| icon_emoji | Emoji to use as the page icon (e.g., '📄', '🚀') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| page_id | ID of the created page. | str |
| page_url | URL of the created page. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Meeting Notes**: Automatically create meeting notes pages from calendar events with template content.

**Task Creation**: Add new entries to a task database when issues are created in other systems.

**Content Publishing**: Create draft pages in a content calendar from AI-generated or imported content.
<!-- END MANUAL -->

---
