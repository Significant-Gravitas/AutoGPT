# Notion Read Page Markdown
<!-- MANUAL: file_description -->
Blocks for reading Notion pages and converting them to Markdown format.
<!-- END MANUAL -->

## Notion Read Page Markdown

### What it is
Read a Notion page and convert it to Markdown format with proper formatting for headings, lists, links, and rich text.

### How it works
<!-- MANUAL: how_it_works -->
This block reads a Notion page and converts its content to Markdown format. It handles Notion's block structure and rich text, translating headings, lists, links, bold, italic, and other formatting into standard Markdown.

The conversion preserves the document structure while making the content portable and usable in other contexts. Optionally include the page title as a top-level header in the output.
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
**Content Export**: Export Notion pages as Markdown for use in static site generators or documentation tools.

**AI Processing**: Convert Notion content to Markdown for LLM processing, summarization, or analysis.

**Cross-Platform Publishing**: Use Notion as a CMS and export content as Markdown for blogs or wikis.
<!-- END MANUAL -->

---
