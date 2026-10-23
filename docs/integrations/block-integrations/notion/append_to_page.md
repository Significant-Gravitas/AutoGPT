# Notion Append To Page
<!-- MANUAL: file_description -->
Blocks for adding content to pages that already exist in Notion.
<!-- END MANUAL -->

## Notion Append To Page

### What it is
Add content to the end of an existing Notion page. Supports markdown content.

### How it works
<!-- MANUAL: how_it_works -->
This block adds content to the end of an existing page through the Notion API. The content can be markdown, which is converted to Notion blocks the same way the Create Page block converts it: headings, bulleted and numbered lists, code blocks, quotes, dividers and paragraphs.

Notion takes at most 100 blocks per request, so longer content is sent in several requests, in order. The block outputs the IDs of the new blocks in page order. A line longer than Notion's 2,000-character limit for a run of text is split into several runs within the same block. If a later request fails, the block outputs only the error, which says how many blocks were added and gives the ID of the last one, so nothing wired to the block's IDs runs on a partial append. The failed request's blocks may or may not have reached the page, so check it before retrying.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| page_id | ID of the page to add content to. Must be accessible by the connected integration. | str | Yes |
| content | Content to add at the end of the page. Can be plain text or markdown - will be converted to Notion blocks. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| block_ids | IDs of the blocks added to the page, in page order. | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Running Logs**: Add each day's summary to the bottom of a shared log page.

**Meeting Notes**: Append action items to a meeting page once a transcript has been processed.

**Research Collection**: Add findings to a page as a workflow gathers them, without rewriting what is already there.
<!-- END MANUAL -->

---
