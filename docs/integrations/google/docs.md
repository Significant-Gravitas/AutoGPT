# Google Docs
<!-- MANUAL: file_description -->
Blocks for creating and editing Google Docs documents.
<!-- END MANUAL -->

## Google Docs Append Markdown

### What it is
Append Markdown content to the end of a Google Doc with full formatting - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
This block appends Markdown content to the end of a Google Doc and automatically converts it to native Google Docs formatting using the Google Docs API. It supports headers, bold, italic, links, lists, and code formatting.

Set add_newline to true to insert a line break before the appended content. The document is returned for chaining with other document operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to append to | Document | No |
| markdown | Markdown content to append to the document | str | Yes |
| add_newline | Add a newline before the appended content | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the append operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**AI Report Generation**: Append LLM-generated analysis or summaries to existing report documents with proper formatting.

**Content Aggregation**: Continuously add formatted content from multiple sources to a running document.

**Meeting Notes**: Append AI-transcribed and formatted meeting notes to shared team documents.
<!-- END MANUAL -->

---

## Google Docs Append Plain Text

### What it is
Append plain text to the end of a Google Doc (no formatting applied)

### How it works
<!-- MANUAL: how_it_works -->
This block appends unformatted text to the end of a Google Doc using the Google Docs API. Unlike the Markdown version, text is inserted exactly as provided without any formatting interpretation.

The block finds the document's end index and inserts the text there, with an optional newline prefix. This is useful for log entries, raw data, or when formatting is handled elsewhere.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to append to | Document | No |
| text | Plain text to append (no formatting applied) | str | Yes |
| add_newline | Add a newline before the appended text | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if append failed | str |
| result | Result of the append operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Activity Logging**: Append timestamped log entries to document-based activity logs.

**Data Capture**: Add raw data or transcript text that will be formatted later.

**Simple Notes**: Quickly add text notes without worrying about formatting.
<!-- END MANUAL -->

---

## Google Docs Create

### What it is
Create a new Google Doc

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new Google Doc in the user's Google Drive using the Google Docs API. You specify a title for the document and optionally provide initial text content.

The newly created document is returned with its ID and URL, allowing immediate access and chaining to other document operations like formatting or sharing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Title for the new document | str | Yes |
| initial_content | Optional initial text content | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if creation failed | str |
| document | The created document | GoogleDriveFile |
| document_id | ID of the created document | str |
| document_url | URL to open the document | str |

### Possible use case
<!-- MANUAL: use_case -->
**Report Templates**: Create new documents for each report cycle with standardized titles.

**Dynamic Document Generation**: Generate personalized documents for customers or projects.

**Workflow Automation**: Create documents as part of onboarding or project kickoff workflows.
<!-- END MANUAL -->

---

## Google Docs Delete Content

### What it is
Delete a range of content from a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
This block removes content from a Google Doc by specifying start and end index positions using the Google Docs API. Index positions are 1-based (index 0 is reserved for a section break).

Use the Get Structure block first to find the correct index positions for content you want to delete. The deletion operation shifts all subsequent content to fill the gap.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| start_index | Start index of content to delete (must be >= 1, as index 0 is a section break) | int | Yes |
| end_index | End index of content to delete | int | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of delete operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Content Cleanup**: Remove outdated sections or placeholder text from templates.

**Document Restructuring**: Delete sections as part of document reorganization workflows.

**Revision Management**: Remove draft content before finalizing documents.
<!-- END MANUAL -->

---

## Google Docs Export

### What it is
Export a Google Doc to PDF, Word, text, or other formats

### How it works
<!-- MANUAL: how_it_works -->
This block exports a Google Doc to various formats (PDF, DOCX, ODT) using the Google Drive API's export functionality. The exported content is returned as base64-encoded data for binary formats.

The export preserves document formatting as closely as possible in the target format. PDF exports are ideal for final distribution, while DOCX exports enable further editing in Microsoft Word.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to export | Document | No |
| format | Export format | "application/pdf" \| "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \| "application/vnd.oasis.opendocument.text" \| "text/plain" \| "text/html" \| "application/epub+zip" \| "application/rtf" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if export failed | str |
| content | Exported content (base64 encoded for binary formats) | str |
| mime_type | MIME type of exported content | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Report Distribution**: Export finalized reports as PDF for email distribution or archival.

**Cross-Platform Sharing**: Export to Word format for recipients who don't use Google Docs.

**Backup Creation**: Create periodic PDF exports of important documents for offline storage.
<!-- END MANUAL -->

---

## Google Docs Find Replace Plain Text

### What it is
Find and replace plain text in a Google Doc (no formatting applied to replacement)

### How it works
<!-- MANUAL: how_it_works -->
This block performs a find-and-replace operation across the entire Google Doc using the Google Docs API. It searches for all occurrences of the specified text and replaces them with the provided replacement text.

The replacement preserves the surrounding formatting but does not apply any new formatting to the replacement text. Case-matching is configurable.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| find_text | Plain text to find | str | Yes |
| replace_text | Plain text to replace with (no formatting applied) | str | Yes |
| match_case | Match case when finding text | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result with replacement count | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Template Population**: Replace placeholder tokens like {{NAME}} with actual values in document templates.

**Batch Updates**: Update company names, dates, or other text across multiple documents.

**Error Correction**: Fix common typos or outdated terminology across documents.
<!-- END MANUAL -->

---

## Google Docs Format Text

### What it is
Apply formatting (bold, italic, color, etc.) to text in a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
This block applies text formatting to a specific range within a Google Doc using the Google Docs API. You specify start and end indexes and choose formatting options like bold, italic, underline, font size, and text color.

Use the Get Structure block to identify the correct index positions. Multiple formatting options can be applied simultaneously in a single request.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| start_index | Start index of text to format (must be >= 1, as index 0 is a section break) | int | Yes |
| end_index | End index of text to format | int | Yes |
| bold | Make text bold | bool | No |
| italic | Make text italic | bool | No |
| underline | Underline text | bool | No |
| font_size | Font size in points (0 = no change) | int | No |
| foreground_color | Text color as hex (e.g., #FF0000 for red) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of format operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Highlight Important Content**: Apply bold or color formatting to emphasize key findings or action items.

**Conditional Formatting**: Format text based on workflow conditions (e.g., red for overdue items).

**Document Styling**: Apply consistent formatting to generated content that matches brand guidelines.
<!-- END MANUAL -->

---

## Google Docs Get Metadata

### What it is
Get metadata about a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves document metadata from a Google Doc using the Google Docs API. It returns information including the document title, unique ID, current revision ID, and the URL for accessing the document.

This metadata is useful for tracking document versions, building document inventories, or generating links for sharing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| title | Document title | str |
| document_id | Document ID | str |
| revision_id | Current revision ID | str |
| document_url | URL to open the document | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Document Inventory**: Gather metadata from multiple documents for tracking and cataloging.

**Version Monitoring**: Track revision IDs to detect when documents have been modified.

**Link Generation**: Extract document URLs for sharing via email or other channels.
<!-- END MANUAL -->

---

## Google Docs Get Structure

### What it is
Get document structure with index positions for precise editing operations

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes a Google Doc's structure and returns detailed information about content segments with their index positions using the Google Docs API. Use flat mode for a simple list of segments or detailed mode for full hierarchical structure.

The index positions are essential for precise editing operations like formatting, deletion, or insertion at specific locations within the document.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to analyze | Document | No |
| detailed | Return full hierarchical structure instead of flat segments | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| segments | Flat list of content segments with indexes (when detailed=False) | List[Dict[str, Any]] |
| structure | Full hierarchical document structure (when detailed=True) | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Position Discovery**: Find correct index positions before performing insert or delete operations.

**Document Analysis**: Understand document structure for content extraction or manipulation.

**Navigation Aid**: Map document sections to enable targeted content operations.
<!-- END MANUAL -->

---

## Google Docs Insert Markdown At

### What it is
Insert formatted Markdown at a specific position in a Google Doc - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
This block inserts Markdown content at a specific index position within a Google Doc, converting the Markdown to native Google Docs formatting using the Google Docs API. Index 1 inserts at the document start.

The Markdown parser handles headers, bold, italic, links, lists, and code formatting. This enables inserting AI-generated content with proper formatting at precise document locations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to insert into | Document | No |
| markdown | Markdown content to insert | str | Yes |
| index | Position index to insert at (1 = start of document) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the insert operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Content Insertion**: Insert AI-generated sections at specific locations in templates.

**Document Assembly**: Build documents by inserting formatted content blocks at designated positions.

**Dynamic Reports**: Insert data-driven formatted content at specific sections of report templates.
<!-- END MANUAL -->

---

## Google Docs Insert Page Break

### What it is
Insert a page break into a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
This block inserts a page break at a specified index position in a Google Doc using the Google Docs API. Setting index to 0 inserts at the end of the document.

Page breaks force subsequent content to start on a new page, useful for separating document sections for printing or PDF generation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| index | Position to insert page break (0 = end of document) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of page break insertion | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Report Formatting**: Add page breaks between major sections of generated reports.

**Print Preparation**: Insert page breaks to control page layout before PDF export.

**Document Structure**: Separate document chapters or sections for better readability.
<!-- END MANUAL -->

---

## Google Docs Insert Plain Text

### What it is
Insert plain text at a specific position in a Google Doc (no formatting applied)

### How it works
<!-- MANUAL: how_it_works -->
This block inserts unformatted text at a specific index position within a Google Doc using the Google Docs API. Index 1 inserts at the document start.

Unlike the Markdown insert, text is inserted exactly as provided without any formatting interpretation, preserving surrounding document formatting.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to insert into | Document | No |
| text | Plain text to insert (no formatting applied) | str | Yes |
| index | Position index to insert at (1 = start of document) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if insert failed | str |
| result | Result of the insert operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Insertion**: Insert raw data values at specific positions in documents.

**Template Variables**: Insert variable values at designated template positions.

**Sequential Content**: Add text entries to specific locations in running documents.
<!-- END MANUAL -->

---

## Google Docs Insert Table

### What it is
Insert a table into a Google Doc, optionally with content and Markdown formatting

### How it works
<!-- MANUAL: how_it_works -->
This block inserts a table into a Google Doc at a specified position using the Google Docs API. You can create empty tables by specifying row/column counts, or provide a 2D array of cell content to create pre-populated tables.

Cell content can optionally be formatted as Markdown, enabling rich formatting like bold headers or links within table cells.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| rows | Number of rows (ignored if content provided) | int | No |
| columns | Number of columns (ignored if content provided) | int | No |
| content | Optional 2D array of cell content, e.g. [['Header1', 'Header2'], ['Row1Col1', 'Row1Col2']]. If provided, rows/columns are derived from this. | List[List[str]] | No |
| index | Position to insert table (0 = end of document) | int | No |
| format_as_markdown | Format cell content as Markdown (headers, bold, links, etc.) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of table insertion | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Presentation**: Insert tables to display structured data from APIs or databases.

**Report Tables**: Add summary tables with metrics, comparisons, or status information.

**Template Tables**: Create table structures that get populated with dynamic content.
<!-- END MANUAL -->

---

## Google Docs Read

### What it is
Read text content from a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
This block extracts the plain text content from a Google Doc using the Google Docs API. It returns the document's text content without formatting information, along with the document title.

Use this for content analysis, text processing, or feeding document content to AI models for summarization or other processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to read | Document | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if read failed | str |
| text | Plain text content of the document | str |
| title | Document title | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Content Extraction**: Read document text for processing, analysis, or AI summarization.

**Search and Index**: Extract text from documents for full-text search indexing.

**Content Migration**: Read document content to transform or migrate to other systems.
<!-- END MANUAL -->

---

## Google Docs Replace All With Markdown

### What it is
Replace entire Google Doc content with formatted Markdown - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
This block clears all existing content from a Google Doc and replaces it with new formatted Markdown content using the Google Docs API. The Markdown is converted to native Google Docs formatting.

This is ideal for completely regenerating document content from AI-generated Markdown output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to replace content in | Document | No |
| markdown | Markdown content to replace the document with | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the replace operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Document Regeneration**: Completely replace document content with newly generated AI output.

**Content Refresh**: Update recurring documents with fresh content while preserving the document.

**Template Reset**: Clear and repopulate template documents for new projects or periods.
<!-- END MANUAL -->

---

## Google Docs Replace Content With Markdown

### What it is
Find text and replace it with formatted Markdown - ideal for LLM/AI output and templates

### How it works
<!-- MANUAL: how_it_works -->
This block finds specific text (like a placeholder token) in a Google Doc and replaces it with formatted Markdown content using the Google Docs API. The Markdown is converted to native Google Docs formatting.

Use this for template systems where placeholders like {{SECTION}} are replaced with AI-generated formatted content.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| find_text | Text to find and replace (e.g., '{{PLACEHOLDER}}' or any text) | str | Yes |
| markdown | Markdown content to replace the found text with | str | Yes |
| match_case | Match case when finding text | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result with replacement count | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Smart Templates**: Replace placeholder tokens with AI-generated formatted content in templates.

**Dynamic Sections**: Populate document sections with contextual formatted content.

**Mail Merge Plus**: Advanced mail merge with formatted content replacement, not just plain text.
<!-- END MANUAL -->

---

## Google Docs Replace Range With Markdown

### What it is
Replace a specific index range in a Google Doc with formatted Markdown - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
This block replaces content between specific start and end index positions with formatted Markdown content using the Google Docs API. The existing content in the range is deleted and replaced with the new formatted content.

Use Get Structure to find the correct index positions. This enables precise replacement of specific document sections with new formatted content.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| markdown | Markdown content to insert in place of the range | str | Yes |
| start_index | Start index of the range to replace (must be >= 1) | int | Yes |
| end_index | End index of the range to replace | int | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the replace operation | Dict[str, Any] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Section Updates**: Replace specific document sections with updated content while preserving the rest.

**Targeted Regeneration**: Regenerate specific portions of documents with new AI-generated content.

**Incremental Updates**: Update identified sections of recurring reports without affecting other areas.
<!-- END MANUAL -->

---

## Google Docs Set Public Access

### What it is
Make a Google Doc public or private

### How it works
<!-- MANUAL: how_it_works -->
This block modifies the sharing permissions of a Google Doc using the Google Drive API to make it publicly accessible or private. You can set the access level to reader (view only) or commenter.

When made public, anyone with the link can access the document according to the specified role. The share link is returned for distribution.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| public | True to make public, False to make private | bool | No |
| role | Permission role for public access | "reader" \| "commenter" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the operation | Dict[str, Any] |
| share_link | Link to the document | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Public Publishing**: Make finalized documents publicly accessible for broad distribution.

**Access Toggle**: Automate switching document access based on workflow stages.

**Link Sharing**: Generate shareable links for documents that don't require individual access grants.
<!-- END MANUAL -->

---

## Google Docs Share

### What it is
Share a Google Doc with specific users

### How it works
<!-- MANUAL: how_it_works -->
This block shares a Google Doc with specific users by email address using the Google Drive API. You can set the permission level (reader, writer, commenter) and optionally send a notification email with a custom message.

Leave the email blank to just generate a shareable link. The block returns the share link for easy distribution.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to share | Document | No |
| email | Email address to share with. Leave empty for link sharing. | str | No |
| role | Permission role for the user | "reader" \| "writer" \| "commenter" | No |
| send_notification | Send notification email to the user | bool | No |
| message | Optional message to include in notification email | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if share failed | str |
| result | Result of the share operation | Dict[str, Any] |
| share_link | Link to the document | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Collaboration**: Share generated documents with stakeholders automatically after creation.

**Workflow Notifications**: Share documents and notify recipients as part of approval workflows.

**Client Delivery**: Share completed deliverables with clients including notification messages.
<!-- END MANUAL -->

---
