# Google Docs Append Markdown

### What it is
Append Markdown content to the end of a Google Doc with full formatting - ideal for LLM/AI output.

### What it does
Append Markdown content to the end of a Google Doc with full formatting - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of the append operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Append Plain Text

### What it is
Append plain text to the end of a Google Doc (no formatting applied).

### What it does
Append plain text to the end of a Google Doc (no formatting applied)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of the append operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Create

### What it is
Create a new Google Doc.

### What it does
Create a new Google Doc

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Delete Content

### What it is
Delete a range of content from a Google Doc.

### What it does
Delete a range of content from a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of delete operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Export

### What it is
Export a Google Doc to PDF, Word, text, or other formats.

### What it does
Export a Google Doc to PDF, Word, text, or other formats

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to export | Document | No |
| format | Export format | "application/pdf" | "application/vnd.openxmlformats-officedocument.wordprocessingml.document" | "application/vnd.oasis.opendocument.text" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if export failed | str |
| content | Exported content (base64 encoded for binary formats) | str |
| mime_type | MIME type of exported content | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Find Replace Plain Text

### What it is
Find and replace plain text in a Google Doc (no formatting applied to replacement).

### What it does
Find and replace plain text in a Google Doc (no formatting applied to replacement)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result with replacement count | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Format Text

### What it is
Apply formatting (bold, italic, color, etc.

### What it does
Apply formatting (bold, italic, color, etc.) to text in a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of format operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Get Metadata

### What it is
Get metadata about a Google Doc.

### What it does
Get metadata about a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Get Structure

### What it is
Get document structure with index positions for precise editing operations.

### What it does
Get document structure with index positions for precise editing operations

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| segments | Flat list of content segments with indexes (when detailed=False) | List[Dict[str, True]] |
| structure | Full hierarchical document structure (when detailed=True) | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Insert Markdown At

### What it is
Insert formatted Markdown at a specific position in a Google Doc - ideal for LLM/AI output.

### What it does
Insert formatted Markdown at a specific position in a Google Doc - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of the insert operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Insert Page Break

### What it is
Insert a page break into a Google Doc.

### What it does
Insert a page break into a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of page break insertion | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Insert Plain Text

### What it is
Insert plain text at a specific position in a Google Doc (no formatting applied).

### What it does
Insert plain text at a specific position in a Google Doc (no formatting applied)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of the insert operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Insert Table

### What it is
Insert a table into a Google Doc, optionally with content and Markdown formatting.

### What it does
Insert a table into a Google Doc, optionally with content and Markdown formatting

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of table insertion | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Read

### What it is
Read text content from a Google Doc.

### What it does
Read text content from a Google Doc

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Replace All With Markdown

### What it is
Replace entire Google Doc content with formatted Markdown - ideal for LLM/AI output.

### What it does
Replace entire Google Doc content with formatted Markdown - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of the replace operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Replace Content With Markdown

### What it is
Find text and replace it with formatted Markdown - ideal for LLM/AI output and templates.

### What it does
Find text and replace it with formatted Markdown - ideal for LLM/AI output and templates

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result with replacement count | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Replace Range With Markdown

### What it is
Replace a specific index range in a Google Doc with formatted Markdown - ideal for LLM/AI output.

### What it does
Replace a specific index range in a Google Doc with formatted Markdown - ideal for LLM/AI output

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| result | Result of the replace operation | Dict[str, True] |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Set Public Access

### What it is
Make a Google Doc public or private.

### What it does
Make a Google Doc public or private

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc | Document | No |
| public | True to make public, False to make private | bool | No |
| role | Permission role for public access | "reader" | "commenter" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the operation | Dict[str, True] |
| share_link | Link to the document | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Docs Share

### What it is
Share a Google Doc with specific users.

### What it does
Share a Google Doc with specific users

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document | Select a Google Doc to share | Document | No |
| email | Email address to share with. Leave empty for link sharing. | str | No |
| role | Permission role for the user | "reader" | "writer" | "commenter" | No |
| send_notification | Send notification email to the user | bool | No |
| message | Optional message to include in notification email | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if share failed | str |
| result | Result of the share operation | Dict[str, True] |
| share_link | Link to the document | str |
| document | The document for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
