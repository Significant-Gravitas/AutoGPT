# Notion Read Page
<!-- MANUAL: file_description -->
Blocks for reading Notion pages and retrieving their raw JSON data.
<!-- END MANUAL -->

## Notion Read Page

### What it is
Read a Notion page by its ID and return its raw JSON.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a Notion page by its ID using the Notion API. The page must be accessible to your connected integration, which requires sharing the page with your integration from within Notion.

The block returns the raw JSON representation of the page, including all properties, metadata, and block IDs. This format is useful for programmatic processing or when you need full access to page data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| page_id | Notion page ID. Must be accessible by the connected integration. You can get this from the page URL notion.so/A-Page-586edd711467478da59fe3ce29a1ffab would be 586edd711467478da59fe35e29a1ffab | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| page | Raw Notion page JSON. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Data Extraction**: Read page properties and metadata for analysis or migration to other systems.

**Automation Triggers**: Check page properties to decide what actions to take in a workflow.

**Content Backup**: Retrieve full page data for archival or backup purposes.
<!-- END MANUAL -->

---
