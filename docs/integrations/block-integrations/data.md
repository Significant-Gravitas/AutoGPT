# Data
<!-- MANUAL: file_description -->
Blocks for creating, reading, and manipulating data structures including lists, dictionaries, spreadsheets, and persistent storage.
<!-- END MANUAL -->

## Create Dictionary

### What it is
Creates a dictionary with the specified key-value pairs. Use this when you know all the values you want to add upfront.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new dictionary from specified key-value pairs in a single operation. It's designed for cases where you know all the data upfront, rather than building the dictionary incrementally.

The block takes a dictionary input and outputs it as-is, making it useful as a starting point for workflows that need to pass structured data between blocks.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| values | Key-value pairs to create the dictionary with | Dict[str, Any] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if dictionary creation failed | str |
| dictionary | The created dictionary containing the specified key-value pairs | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**API Request Payloads**: Create complete request body objects with all required fields before sending to an API.

**Configuration Objects**: Build settings dictionaries with predefined values for initializing services or workflows.

**Data Mapping**: Transform input data into a structured format with specific keys expected by downstream blocks.
<!-- END MANUAL -->

---

## Create List

### What it is
Creates a list with the specified values. Use this when you know all the values you want to add upfront. This block can also yield the list in batches based on a maximum size or token limit.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a list from provided values and can optionally chunk it into smaller batches. When max_size is set, the list is yielded in chunks of that size. When max_tokens is set, chunks are sized to fit within token limits for LLM processing.

This batching capability is particularly useful when processing large datasets that need to be split for API limits or memory constraints.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| values | A list of values to be combined into a new list. | List[Any] | Yes |
| max_size | Maximum size of the list. If provided, the list will be yielded in chunks of this size. | int | No |
| max_tokens | Maximum tokens for the list. If provided, the list will be yielded in chunks that fit within this token limit. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| list | The created list containing the specified values. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Batch Processing**: Split large datasets into manageable chunks for API calls with rate limits.

**LLM Token Management**: Divide text content into token-limited batches for processing by language models.

**Parallel Processing**: Create batches of work items that can be processed concurrently by multiple blocks.
<!-- END MANUAL -->

---

## File Read

### What it is
Reads a file and returns its content as a string, with optional chunking by delimiter and size limits

### How it works
<!-- MANUAL: how_it_works -->
This block reads file content from various sources (URL, data URI, or local path) and returns it as a string. It supports chunking via delimiter (like newlines) or size limits, yielding content in manageable pieces.

Use skip_rows and skip_size to skip header content or initial bytes. When delimiter and limits are set, content is yielded chunk by chunk, enabling processing of large files without loading everything into memory.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_input | The file to read from (URL, data URI, or local path) | str (file) | Yes |
| delimiter | Delimiter to split the content into rows/chunks (e.g., '\n' for lines) | str | No |
| size_limit | Maximum size in bytes per chunk to yield (0 for no limit) | int | No |
| row_limit | Maximum number of rows to process (0 for no limit, requires delimiter) | int | No |
| skip_size | Number of characters to skip from the beginning of the file | int | No |
| skip_rows | Number of rows to skip from the beginning (requires delimiter) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content | File content, yielded as individual chunks when delimiter or size limits are applied | str |

### Possible use case
<!-- MANUAL: use_case -->
**Log File Processing**: Read and process log files line by line, filtering or transforming each entry.

**Large Document Analysis**: Read large text files in chunks for summarization or analysis without memory issues.

**Data Import**: Read text-based data files and process them row by row for database import.
<!-- END MANUAL -->

---

## Persist Information

### What it is
Persist key-value information for the current user

### How it works
<!-- MANUAL: how_it_works -->
This block stores key-value data that persists across workflow runs. You can scope the persistence to either within_agent (available to all runs of this specific agent) or across_agents (available to all agents for this user).

The stored data remains available until explicitly overwritten, enabling state management and configuration persistence between workflow executions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| key | Key to store the information under | str | Yes |
| value | Value to store | Value | Yes |
| scope | Scope of persistence: within_agent (shared across all runs of this agent) or across_agents (shared across all agents for this user) | "within_agent" \| "across_agents" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| value | Value that was stored | Value |

### Possible use case
<!-- MANUAL: use_case -->
**User Preferences**: Store user settings like preferred language or notification preferences for future runs.

**Progress Tracking**: Save the last processed item ID to resume batch processing where you left off.

**API Token Caching**: Store refreshed API tokens that can be reused across multiple workflow executions.
<!-- END MANUAL -->

---

## Read Spreadsheet

### What it is
Reads CSV and Excel files and outputs the data as a list of dictionaries and individual rows. Excel files are automatically converted to CSV format.

### How it works
<!-- MANUAL: how_it_works -->
This block parses CSV and Excel files, converting each row into a dictionary with column headers as keys. Excel files are automatically converted to CSV format before processing.

Configure delimiter, quote character, and escape character for proper CSV parsing. Use skip_rows to ignore headers or initial rows, and skip_columns to exclude unwanted columns from the output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| contents | The contents of the CSV/spreadsheet data to read | str | No |
| file_input | CSV or Excel file to read from (URL, data URI, or local path). Excel files are automatically converted to CSV | str (file) | No |
| delimiter | The delimiter used in the CSV/spreadsheet data | str | No |
| quotechar | The character used to quote fields | str | No |
| escapechar | The character used to escape the delimiter | str | No |
| has_header | Whether the CSV file has a header row | bool | No |
| skip_rows | The number of rows to skip from the start of the file | int | No |
| strip | Whether to strip whitespace from the values | bool | No |
| skip_columns | The columns to skip from the start of the row | List[str] | No |
| produce_singular_result | If True, yield individual 'row' outputs only (can be slow). If False, yield both 'rows' (all data) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| row | The data produced from each row in the spreadsheet | Dict[str, str] |
| rows | All the data in the spreadsheet as a list of rows | List[Dict[str, str]] |

### Possible use case
<!-- MANUAL: use_case -->
**Data Import**: Import product catalogs, contact lists, or inventory data from spreadsheet exports.

**Report Processing**: Parse generated CSV reports from other systems for analysis or transformation.

**Bulk Operations**: Process spreadsheets of email addresses, user records, or configuration data row by row.
<!-- END MANUAL -->

---

## Retrieve Information

### What it is
Retrieve key-value information for the current user

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves previously stored key-value data for the current user. Specify the key and scope to fetch the corresponding value. If the key doesn't exist, the default_value is returned.

Use within_agent scope for agent-specific data or across_agents for data shared across all user agents.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| key | Key to retrieve the information for | str | Yes |
| scope | Scope of persistence: within_agent (shared across all runs of this agent) or across_agents (shared across all agents for this user) | "within_agent" \| "across_agents" | No |
| default_value | Default value to return if key is not found | Default Value | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| value | Retrieved value or default value | Value |

### Possible use case
<!-- MANUAL: use_case -->
**Resume Processing**: Retrieve the last processed item ID to continue batch operations from where you left off.

**Load Preferences**: Fetch stored user preferences at workflow start to customize behavior.

**State Restoration**: Retrieve workflow state saved from a previous run to maintain continuity.
<!-- END MANUAL -->

---

## Screenshot Web Page

### What it is
Takes a screenshot of a specified website using ScreenshotOne API

### How it works
<!-- MANUAL: how_it_works -->
This block uses the ScreenshotOne API to capture screenshots of web pages. Configure viewport dimensions, output format, and whether to capture the full page or just the visible area.

Optional features include blocking ads, cookie banners, and chat widgets for cleaner screenshots. Caching can be enabled to improve performance for repeated captures of the same page.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | URL of the website to screenshot | str | Yes |
| viewport_width | Width of the viewport in pixels | int | No |
| viewport_height | Height of the viewport in pixels | int | No |
| full_page | Whether to capture the full page length | bool | No |
| format | Output format (png, jpeg, webp) | "png" \| "jpeg" \| "webp" | No |
| block_ads | Whether to block ads | bool | No |
| block_cookie_banners | Whether to block cookie banners | bool | No |
| block_chats | Whether to block chat widgets | bool | No |
| cache | Whether to enable caching | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| image | The screenshot image data | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**Visual Documentation**: Capture screenshots of web pages for documentation, reports, or archives.

**Competitive Monitoring**: Regularly screenshot competitor websites to track design and content changes.

**Visual Testing**: Capture page renders for visual regression testing or design verification workflows.
<!-- END MANUAL -->

---
