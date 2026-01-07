# Notion Read Page Markdown

### What it is
Read a Notion page and convert it to Markdown format with proper formatting for headings, lists, links, and rich text.

### What it does
Read a Notion page and convert it to Markdown format with proper formatting for headings, lists, links, and rich text.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| page_id | Notion page ID. Must be accessible by the connected integration. You can get this from the page URL notion.so/A-Page-586edd711467478da59fe35e29a1ffab would be 586edd711467478da59fe35e29a1ffab | str | Yes |
| include_title | Whether to include the page title as a header in the markdown | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| markdown | Page content in Markdown format. | str |
| title | Page title. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
