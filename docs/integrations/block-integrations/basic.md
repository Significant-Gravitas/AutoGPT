# Basic
<!-- MANUAL: file_description -->
Core utility blocks for storing values, printing output, file operations, type conversion, and basic data manipulation.
<!-- END MANUAL -->

## Add Memory

### What it is
Add new memories to Mem0 with user segmentation

### How it works
<!-- MANUAL: how_it_works -->
This block integrates with Mem0, a memory layer service that stores and retrieves information across conversations. When you add a memory, the content is stored with the user's context and can optionally be segmented by run or agent, allowing for scoped memory retrieval later.

The block accepts either plain text or structured message objects (like those from AI blocks). You can attach metadata to memories for better organization and filtering. Memories persist across workflow executions, enabling your agents to "remember" past interactions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| content | Content to add - either a string or list of message objects as output from an AI block | Content | No |
| metadata | Optional metadata for the memory | Dict[str, Any] | No |
| limit_memory_to_run | Limit the memory to the run | bool | No |
| limit_memory_to_agent | Limit the memory to the agent | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| action | Action of the operation | str |
| memory | Memory created | str |
| results | List of all results from the operation | List[Dict[str, str]] |

### Possible use case
<!-- MANUAL: use_case -->
**Personalized Assistants**: Store user preferences, past interactions, or learned information so your AI agent can provide personalized responses in future conversations.

**Context Carryover**: Save important details from one workflow run (like customer issues or project context) to reference in subsequent runs without asking the user again.

**Knowledge Building**: Accumulate facts and insights over time, creating a growing knowledge base that improves your agent's helpfulness with each interaction.
<!-- END MANUAL -->

---

## Add To Dictionary

### What it is
Adds a new key-value pair to a dictionary. If no dictionary is provided, a new one is created.

### How it works
<!-- MANUAL: how_it_works -->
This block adds one or more key-value pairs to a dictionary. If you don't provide an existing dictionary, it creates a new one. You can add entries one at a time using the key/value fields, or add multiple entries at once using the entries field.

The block outputs the updated dictionary with all new entries added. This is useful for building up structured data objects as your workflow progresses, collecting information from multiple sources into a single data structure.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to add the entry to. If not provided, a new dictionary will be created. | Dict[str, Any] | No |
| key | The key for the new entry. | str | No |
| value | The value for the new entry. | Value | No |
| entries | The entries to add to the dictionary. This is the batch version of the `key` and `value` fields. | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_dictionary | The dictionary with the new entry added. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Building API Payloads**: Construct complex JSON objects by adding fields from different workflow branches before sending to an API.

**Aggregating Form Data**: Collect user inputs from multiple form fields into a single structured object for processing or storage.

**Creating Configuration Objects**: Build up settings or configuration dictionaries dynamically based on conditional logic in your workflow.
<!-- END MANUAL -->

---

## Add To List

### What it is
Adds a new entry to a list. The entry can be of any type. If no list is provided, a new one is created.

### How it works
<!-- MANUAL: how_it_works -->
This block appends items to a list or creates a new list if none is provided. You can add a single entry or multiple entries at once. The optional position parameter lets you insert items at a specific index rather than appending to the end.

Items can be of any type—strings, numbers, dictionaries, or other lists. This flexibility makes the block useful for building up collections of data as your workflow processes multiple items or accumulates results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list | The list to add the entry to. If not provided, a new list will be created. | List[Any] | No |
| entry | The entry to add to the list. Can be of any type (string, int, dict, etc.). | Entry | No |
| entries | The entries to add to the list. This is the batch version of the `entry` field. | List[Any] | No |
| position | The position to insert the new entry. If not provided, the entry will be appended to the end of the list. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_list | The list with the new entry added. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Collecting Search Results**: Accumulate items from paginated API responses into a single list for batch processing or display.

**Building Email Recipients**: Gather email addresses from various sources into a recipient list before sending a message.

**Aggregating Errors**: Collect validation errors or warnings from multiple checks into a list for consolidated error reporting.
<!-- END MANUAL -->

---

## Agent Date Input

### What it is
Block for date input.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a date picker input field for users interacting with your agent. When the agent runs, users see a calendar widget to select a date, which is then passed to your workflow in YYYY-MM-DD format.

The block is part of the Agent Input family, allowing you to collect structured input from users at runtime rather than hardcoding values. This makes your agents interactive and reusable for different scenarios.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Date input (YYYY-MM-DD). | str (date) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Date result. | str (date) |

### Possible use case
<!-- MANUAL: use_case -->
**Appointment Scheduling**: Let users select a date for booking appointments, meetings, or reservations through your agent.

**Report Generation**: Allow users to specify a date range start or end point for generating custom reports.

**Deadline Setting**: Enable users to set due dates for tasks or projects when creating them through your workflow.
<!-- END MANUAL -->

---

## Agent Dropdown Input

### What it is
Block for dropdown text selection.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a dropdown selection input for users interacting with your agent. You define the available options using placeholder_values, and users select one option from the list at runtime.

This is ideal when you want to constrain user input to a predefined set of choices, ensuring valid input and simplifying the user experience. The selected value is passed to downstream blocks in your workflow.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Text selected from a dropdown. | str | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| placeholder_values | Possible values for the dropdown. | List[Any] | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Selected dropdown value. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Service Selection**: Let users choose from available service tiers (Basic, Pro, Enterprise) when configuring their request.

**Language Selection**: Allow users to select their preferred language from a list of supported options.

**Category Filtering**: Enable users to select a category (Sales, Support, Billing) to route their inquiry appropriately.
<!-- END MANUAL -->

---

## Agent File Input

### What it is
Block for file upload input (string path for example).

### How it works
<!-- MANUAL: how_it_works -->
This block provides a file upload input for users interacting with your agent. Users can upload files which are stored temporarily and passed to your workflow as a file path reference.

By default, the block outputs a file path string that other blocks can use to access the uploaded file. The optional base64 mode converts the file content to base64 encoding, though using file paths is generally recommended for better performance with large files.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Path or reference to an uploaded file. | str (file) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |
| base_64 | Whether produce an output in base64 format (not recommended, you can pass the string path just fine accross blocks). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | File reference/path result. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Document Processing**: Accept PDF or Word documents from users for analysis, summarization, or data extraction.

**Image Upload**: Allow users to upload images for processing, resizing, or AI-based analysis.

**Data Import**: Enable users to upload CSV or Excel files to import data into your workflow for processing.
<!-- END MANUAL -->

---

## Agent Google Drive File Input

### What it is
Block for selecting a file from Google Drive.

### How it works
<!-- MANUAL: how_it_works -->
This block integrates with Google Drive to let users select files directly from their connected Drive account. The Google Drive file picker appears at runtime, allowing users to browse and select files without manually copying file IDs or URLs.

You can configure which file types to display (documents, spreadsheets, presentations) using the allowed_views option. The block outputs the selected file's metadata including its ID, name, and URL for use by other Google-integrated blocks.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The selected Google Drive file. | GoogleDriveFile | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |
| allowed_views | Which views to show in the file picker (DOCS, SPREADSHEETS, PRESENTATIONS, etc.). | List["DOCS" \| "DOCUMENTS" \| "SPREADSHEETS" \| "PRESENTATIONS" \| "DOCS_IMAGES" \| "FOLDERS"] | No |
| allow_folder_selection | Whether to allow selecting folders. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | The selected Google Drive file with ID, name, URL, and other metadata. | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Document Workflow**: Let users select a Google Doc to read, analyze, or append content to without knowing the file ID.

**Spreadsheet Data Import**: Allow users to pick a Google Sheet to import data from for processing or analysis.

**File Organization**: Enable users to select folders or files for bulk operations like moving, copying, or organizing content.
<!-- END MANUAL -->

---

## Agent Input

### What it is
A block that accepts and processes user input values within a workflow, supporting various input types and validation.

### How it works
<!-- MANUAL: how_it_works -->
It accepts a value from the user, along with metadata such as name, description, and optional placeholder values. The block then outputs the provided value.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The value to be passed as input. | Value | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | The value passed as input. | Result |

### Possible use case
<!-- MANUAL: use_case -->
Collecting user preferences at the start of a personalized recommendation workflow.
<!-- END MANUAL -->

---

## Agent Long Text Input

### What it is
Block for long text input (multi-line).

### How it works
<!-- MANUAL: how_it_works -->
This block provides a multi-line text area input for users interacting with your agent. Unlike the short text input, this displays a larger text area suitable for paragraphs, descriptions, or any content that may span multiple lines.

The block is ideal for collecting longer-form content like messages, descriptions, or code snippets from users at runtime. The text is passed as-is to downstream blocks in your workflow.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Long text input (potentially multi-line). | str (long-text) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Long text result. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Feedback Collection**: Accept detailed user feedback, reviews, or comments that may require multiple paragraphs.

**Content Submission**: Let users submit articles, blog posts, or documentation content for processing or publication.

**Query Input**: Allow users to enter complex questions or prompts for AI processing that require detailed context.
<!-- END MANUAL -->

---

## Agent Number Input

### What it is
Block for number input.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a numeric input field for users interacting with your agent. The input validates that the user enters a valid integer, preventing text or invalid values from being submitted.

This is useful when you need numeric parameters like quantities, counts, limits, or any integer value from users at runtime. The number is passed to downstream blocks for use in calculations or configurations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Number input. | int | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Number result. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Quantity Selection**: Let users specify how many items to process, order, or generate.

**Pagination Control**: Allow users to specify page numbers or result limits for data retrieval.

**Threshold Setting**: Enable users to set numeric thresholds or limits for alerts, filtering, or processing logic.
<!-- END MANUAL -->

---

## Agent Output

### What it is
A block that records and formats workflow results for display to users, with optional Jinja2 template formatting support.

### How it works
<!-- MANUAL: how_it_works -->
It accepts an input value along with a name, description, and optional format string. If a format string is provided, it attempts to apply the formatting to the input value before outputting it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| value | The value to be recorded as output. | Value | No |
| name | The name of the output. | str | Yes |
| title | The title of the output. | str | No |
| description | The description of the output. | str | No |
| format | The format string to be used to format the recorded_value. Use Jinja2 syntax. | str | No |
| escape_html | Whether to escape special characters in the inserted values to be HTML-safe. Enable for HTML output, disable for plain text. | bool | No |
| advanced | Whether to treat the output as advanced. | bool | No |
| secret | Whether the output should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| output | The value recorded as output. | Output |
| name | The name of the value recorded as output. | Name |

### Possible use case
<!-- MANUAL: use_case -->
Presenting the final results of a data analysis workflow in a specific format.
<!-- END MANUAL -->

---

## Agent Short Text Input

### What it is
Block for short text input (single-line).

### How it works
<!-- MANUAL: how_it_works -->
This block provides a single-line text input field for users interacting with your agent. It's designed for brief text entries like names, titles, URLs, or short responses.

The input displays as a standard text field and passes the entered text to downstream blocks. Use this for collecting concise information that doesn't require multiple lines or extensive formatting.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Short text input. | str (short-text) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Short text result. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Name Collection**: Gather user names, company names, or project names for personalization.

**Search Queries**: Accept search terms or keywords from users to drive search functionality.

**URL Input**: Let users provide URLs for websites, APIs, or resources to process in your workflow.
<!-- END MANUAL -->

---

## Agent Table Input

### What it is
Block for table data input with customizable headers.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a tabular data input interface for users interacting with your agent. Users can enter data in a spreadsheet-like table format with customizable column headers.

The table input is ideal for structured data entry where users need to provide multiple records with consistent fields. The block outputs the data as a list of dictionaries, with each row becoming a dictionary where column headers are keys.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The table data as a list of dictionaries. | List[Dict[str, Any]] | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |
| column_headers | Column headers for the table. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | The table data as a list of dictionaries with headers as keys. | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Bulk Data Entry**: Let users input multiple records at once, like a list of contacts with name, email, and phone columns.

**Order Processing**: Accept line items for an order with product, quantity, and price columns.

**Task Lists**: Allow users to submit multiple tasks with columns for title, assignee, and priority.
<!-- END MANUAL -->

---

## Agent Time Input

### What it is
Block for time input.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a time picker input for users interacting with your agent. Users select a time (hours, minutes, and optionally seconds) which is passed to your workflow in HH:MM:SS format.

The time picker provides a user-friendly interface for selecting times without requiring users to type in a specific format. This ensures valid time values and improves the user experience.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Time input (HH:MM:SS). | str (time) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Time result. | str (time) |

### Possible use case
<!-- MANUAL: use_case -->
**Appointment Scheduling**: Let users specify a time for meetings, calls, or appointments.

**Reminder Setting**: Allow users to set reminder times for notifications or alerts.

**Shift Configuration**: Enable users to define start or end times for work shifts or availability windows.
<!-- END MANUAL -->

---

## Agent Toggle Input

### What it is
Block for boolean toggle input.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a boolean toggle (on/off switch) input for users interacting with your agent. Users simply click to toggle between true and false states, making yes/no decisions quick and error-free.

The toggle is ideal for binary choices like enabling features, confirming actions, or setting preferences. The boolean value is passed to downstream blocks for conditional logic.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Boolean toggle input. | bool | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Boolean toggle result. | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Feature Flags**: Let users enable or disable optional features in your workflow.

**Confirmation Toggles**: Require users to acknowledge terms, confirm destructive actions, or opt into notifications.

**Mode Selection**: Allow users to switch between modes like "test mode" vs "production mode" or "verbose" vs "quiet" output.
<!-- END MANUAL -->

---

## Block Installation

### What it is
Given a code string, this block allows the verification and installation of a block code into the system.

### How it works
<!-- MANUAL: how_it_works -->
This block allows dynamic installation of new block types into the system from Python code. The code is verified for safety and correctness before installation. Once installed, the new block becomes available for use in workflows.

This enables extensibility by allowing custom blocks to be added without modifying the core system, though it requires the code to follow the block specification format.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| code | Python code of the block to be installed | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the block installation fails | str |
| success | Success message if the block is installed successfully | str |

### Possible use case
<!-- MANUAL: use_case -->
**Custom Integrations**: Install blocks that connect to proprietary or internal APIs not covered by built-in blocks.

**Dynamic Workflows**: Allow administrators to extend workflow capabilities without redeploying the entire system.

**Experimental Features**: Test new block implementations before formally adding them to the block library.
<!-- END MANUAL -->

---

## Concatenate Lists

### What it is
Concatenates multiple lists into a single list. All elements from all input lists are combined in order.

### How it works
<!-- MANUAL: how_it_works -->
The block iterates through each list in the input and extends a result list with all elements from each one. It processes lists in order, so `[[1, 2], [3, 4]]` becomes `[1, 2, 3, 4]`.

The block includes validation to ensure each item is actually a list. If a non-list value (like a string or number) is encountered, the block outputs an error message instead of proceeding. None values are skipped automatically.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| lists | A list of lists to concatenate together. All lists will be combined in order into a single list. | List[List[Any]] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if concatenation failed due to invalid input types. | str |
| concatenated_list | The concatenated list containing all elements from all input lists in order. | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Paginated API Merging**: Combine results from multiple API pages into a single list for batch processing or display.

**Parallel Task Aggregation**: Merge outputs from parallel workflow branches that each produce a list of results.

**Multi-Source Data Collection**: Combine data collected from different sources (like multiple RSS feeds or API endpoints) into one unified list.
<!-- END MANUAL -->

---

## Dictionary Is Empty

### What it is
Checks if a dictionary is empty.

### How it works
<!-- MANUAL: how_it_works -->
This block checks whether a dictionary has any entries and returns a boolean result. An empty dictionary (no key-value pairs) returns true, while a dictionary with any entries returns false.

This is useful for conditional logic where you need to verify if data was returned from an API, if user input was provided, or if a collection process yielded any results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to check. | Dict[str, Any] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| is_empty | True if the dictionary is empty. | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Response Validation**: Check if an API returned an empty response before processing data.

**Input Verification**: Verify that user-provided form data contains at least one field before submission.

**Conditional Processing**: Skip processing steps when no matching data was found in a search or filter operation.
<!-- END MANUAL -->

---

## File Store

### What it is
Stores the input file in the temporary directory.

### How it works
<!-- MANUAL: how_it_works -->
This block takes a file from various sources (URL, data URI, or local path) and stores it in a temporary directory for use by other blocks in your workflow. This normalizes file handling regardless of the original source.

The block outputs a file path that other blocks can use to access the stored file. The optional base64 output mode is available but file paths are recommended for better performance with large files.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_in | The file to store in the temporary directory, it can be a URL, data URI, or local path. | str (file) | Yes |
| base_64 | Whether produce an output in base64 format (not recommended, you can pass the string path just fine accross blocks). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file_out | The relative path to the stored file in the temporary directory. | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**URL File Download**: Fetch a file from a URL and make it available for local processing by other blocks.

**Data URI Conversion**: Convert base64-encoded data URIs (like from a web form) into accessible file paths.

**File Normalization**: Standardize file access across different input sources (URLs, uploads, local files) for consistent downstream processing.
<!-- END MANUAL -->

---

## Find In Dictionary

### What it is
A block that looks up a value in a dictionary, list, or object by key or index and returns the corresponding value.

### How it works
<!-- MANUAL: how_it_works -->
This block extracts a value from a dictionary (object) or list using a key or index. If the key exists, the value is output through the "output" pin. If the key is missing, the original input is output through the "missing" pin.

This enables safe data access with built-in handling for missing keys, preventing workflow errors when expected data isn't present. You can use string keys for dictionaries or integer indices for lists.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input | Dictionary to lookup from | Input | Yes |
| key | Key to lookup in the dictionary | str \| int | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | Value found for the given key | Output |
| missing | Value of the input that missing the key | Missing |

### Possible use case
<!-- MANUAL: use_case -->
**API Response Parsing**: Extract specific fields (like "data" or "results") from API response objects.

**Configuration Access**: Retrieve settings from a configuration dictionary by key name.

**User Data Extraction**: Pull specific user attributes (name, email, preferences) from a user profile object.
<!-- END MANUAL -->

---

## Find In List

### What it is
Finds the index of the value in the list.

### How it works
<!-- MANUAL: how_it_works -->
This block searches a list for a specific value and returns its position (index). If found, it outputs the zero-based index and sets "found" to true. If not found, it outputs the original value through "not_found_value" and sets "found" to false.

This enables conditional logic based on list membership and helps locate items for subsequent list operations like replacement or removal.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list | The list to search in. | List[Any] | Yes |
| value | The value to search for. | Value | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| index | The index of the value in the list. | int |
| found | Whether the value was found in the list. | bool |
| not_found_value | The value that was not found in the list. | Not Found Value |

### Possible use case
<!-- MANUAL: use_case -->
**Duplicate Detection**: Check if an item already exists in a list before adding it.

**Status Lookup**: Find if a value is in a list of valid states or allowed values.

**Position Finding**: Locate an item's position for subsequent operations like updates or removals.
<!-- END MANUAL -->

---

## Get All Memories

### What it is
Retrieve all memories from Mem0 with optional conversation filtering

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all stored memories from Mem0 for the current user context. You can filter results by categories or metadata, and scope the retrieval to the current run or agent using the limit options.

Memories are returned as a list that your workflow can iterate through. This is useful for reviewing accumulated knowledge, debugging what your agent has learned, or aggregating past interactions for analysis.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | An unused field that is used to trigger the block when you have no other inputs | bool | No |
| categories | Filter by categories | List[str] | No |
| metadata_filter | Optional metadata filters to apply | Dict[str, Any] | No |
| limit_memory_to_run | Limit the memory to the run | bool | No |
| limit_memory_to_agent | Limit the memory to the agent | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| memories | List of memories | Memories |

### Possible use case
<!-- MANUAL: use_case -->
**Context Review**: Retrieve all memories at the start of a session to understand what your agent already knows about a user.

**Memory Export**: Collect all stored memories for backup, analysis, or migration to another system.

**Memory Management**: List all memories to identify outdated or incorrect information that needs updating.
<!-- END MANUAL -->

---

## Get Latest Memory

### What it is
Retrieve the latest memory from Mem0 with optional key filtering

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the most recently stored memory from Mem0. You can filter by categories, metadata, or conversation ID to find the latest relevant memory. The block indicates whether a memory was found and returns it if available.

This is useful for quickly accessing the last piece of information stored without iterating through all memories, such as checking the most recent user preference or the last conversation topic.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | An unused field that is used to trigger the block when you have no other inputs | bool | No |
| categories | Filter by categories | List[str] | No |
| conversation_id | Optional conversation ID to retrieve the latest memory from (uses run_id) | str | No |
| metadata_filter | Optional metadata filters to apply | Dict[str, Any] | No |
| limit_memory_to_run | Limit the memory to the run | bool | No |
| limit_memory_to_agent | Limit the memory to the agent | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| memory | Latest memory if found | Dict[str, Any] |
| found | Whether a memory was found | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Conversation Continuity**: Retrieve the last topic discussed to provide context when resuming a conversation.

**Status Tracking**: Get the most recent status update or progress report stored during a workflow.

**Quick Recall**: Access the last user preference or setting without loading the full memory history.
<!-- END MANUAL -->

---

## Get List Item

### What it is
Returns the element at the given index.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves an item from a list at a specific index position. It uses zero-based indexing (first item is 0) and supports negative indices for accessing items from the end (e.g., -1 for the last item).

If the index is out of range, the block outputs an error. This is useful for accessing specific elements without iterating through the entire list.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list | The list to get the item from. | List[Any] | Yes |
| index | The 0-based index of the item (supports negative indices). | int | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| item | The item at the specified index. | Item |

### Possible use case
<!-- MANUAL: use_case -->
**First/Last Item Access**: Get the first item (index 0) or last item (index -1) from a list of results.

**Ordered Selection**: Access a specific position in a ranked list, like the second-highest score or third most recent entry.

**Array Unpacking**: Extract individual elements from a fixed-structure list where each position has a known meaning.
<!-- END MANUAL -->

---

## Get Weather Information

### What it is
Retrieves weather information for a specified location using OpenWeatherMap API.

### How it works
<!-- MANUAL: how_it_works -->
The block sends a request to a weather API (like OpenWeatherMap) with the provided location. It then processes the response to extract relevant weather data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| location | Location to get weather information for | str | Yes |
| use_celsius | Whether to use Celsius or Fahrenheit for temperature | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the weather information cannot be retrieved | str |
| temperature | Temperature in the specified location | str |
| humidity | Humidity in the specified location | str |
| condition | Weather condition in the specified location | str |

### Possible use case
<!-- MANUAL: use_case -->
A travel planning application could use this block to provide users with current weather information for their destination cities.
<!-- END MANUAL -->

---

## Human In The Loop

### What it is
Pause execution and wait for human approval or modification of data

### How it works
<!-- MANUAL: how_it_works -->
This block pauses workflow execution and presents data to a human reviewer for approval. The workflow waits until the human approves or rejects the data, then routes to the corresponding output. If editable is enabled, the reviewer can modify the data before approving.

This enables human oversight at critical points in automated workflows, ensuring important decisions have human verification before proceeding.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| data | The data to be reviewed by a human user | Data | Yes |
| name | A descriptive name for what this data represents | str | Yes |
| editable | Whether the human reviewer can edit the data | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| approved_data | The data when approved (may be modified by reviewer) | Approved Data |
| rejected_data | The data when rejected (may be modified by reviewer) | Rejected Data |
| review_message | Any message provided by the reviewer | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Moderation**: Review AI-generated content before publishing to ensure quality and appropriateness.

**Approval Workflows**: Require manager approval for actions like large purchases, access requests, or configuration changes.

**Quality Assurance**: Let reviewers verify data transformations or calculations before they're committed to production systems.
<!-- END MANUAL -->

---

## List Is Empty

### What it is
Checks if a list is empty.

### How it works
<!-- MANUAL: how_it_works -->
This block checks whether a list contains any items and returns a boolean result. An empty list (no elements) returns true, while a list with any elements returns false.

This is useful for conditional logic where you need to verify if search results were found, if items are available for processing, or if a collection has any entries to iterate over.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list | The list to check. | List[Any] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| is_empty | True if the list is empty. | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Search Result Handling**: Check if a search returned any results before processing, displaying "no results found" when empty.

**Batch Processing Guard**: Verify that a list has items before starting a batch operation to avoid empty iterations.

**Conditional Messaging**: Send different notifications based on whether pending items exist or the queue is empty.
<!-- END MANUAL -->

---

## Note

### What it is
A visual annotation block that displays a sticky note in the workflow editor for documentation and organization purposes.

### How it works
<!-- MANUAL: how_it_works -->
It simply accepts a text input and passes it through as an output to be displayed as a note.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to display in the sticky note. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | The text to display in the sticky note. | str |

### Possible use case
<!-- MANUAL: use_case -->
Adding explanatory notes or reminders within a complex workflow to help users understand different stages or provide additional context.
<!-- END MANUAL -->

---

## Print To Console

### What it is
A debugging block that outputs text to the console for monitoring and troubleshooting workflow execution.

### How it works
<!-- MANUAL: how_it_works -->
This block outputs the provided data to the server console log and passes it through as output. It's primarily used for debugging workflows by allowing you to inspect values at any point in the data flow.

The block accepts any data type and both prints it for debugging visibility and forwards it to downstream blocks, making it easy to insert into existing connections without disrupting the workflow.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The data to print to the console. | Text | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | The data printed to the console. | Output |
| status | The status of the print operation. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Workflow Debugging**: Insert at any point to inspect what data is flowing through that connection during testing.

**Variable Inspection**: Log the values of variables or API responses to understand what your workflow is receiving.

**Progress Tracking**: Add print statements at key stages to monitor workflow progress in the server logs.
<!-- END MANUAL -->

---

## Remove From Dictionary

### What it is
Removes a key-value pair from a dictionary.

### How it works
<!-- MANUAL: how_it_works -->
This block removes a key-value pair from a dictionary by specifying the key. The updated dictionary without that entry is output. Optionally, you can retrieve the value that was removed by enabling return_value.

If the key doesn't exist in the dictionary, the operation may error or return the dictionary unchanged depending on the implementation. This is useful for cleaning up data or extracting and removing values in one step.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to modify. | Dict[str, Any] | Yes |
| key | Key to remove from the dictionary. | str \| int | Yes |
| return_value | Whether to return the removed value. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_dictionary | The dictionary after removal. | Dict[str, Any] |
| removed_value | The removed value if requested. | Removed Value |

### Possible use case
<!-- MANUAL: use_case -->
**Data Cleaning**: Remove sensitive fields (like passwords or tokens) from data before logging or storing.

**Pop Pattern**: Extract and remove a value from a dictionary in a single operation, like dequeuing items.

**Object Trimming**: Remove unnecessary or deprecated fields from configuration objects before processing.
<!-- END MANUAL -->

---

## Remove From List

### What it is
Removes an item from a list by value or index.

### How it works
<!-- MANUAL: how_it_works -->
This block removes an item from a list either by value (remove first occurrence) or by index (remove at specific position). Negative indices are supported for removal from the end. Optionally, the removed item can be returned.

This provides flexibility for both "remove this specific item" and "remove the item at this position" use cases in a single block.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list | The list to modify. | List[Any] | Yes |
| value | Value to remove from the list. | Value | No |
| index | Index of the item to pop (supports negative indices). | int | No |
| return_item | Whether to return the removed item. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_list | The list after removal. | List[Any] |
| removed_item | The removed item if requested. | Removed Item |

### Possible use case
<!-- MANUAL: use_case -->
**Queue Processing**: Pop items from the front of a list to process them one at a time (FIFO queue).

**Exclusion Lists**: Remove specific values from a list, like filtering out certain options or invalid entries.

**Stack Operations**: Pop items from the end of a list for last-in-first-out processing.
<!-- END MANUAL -->

---

## Replace Dictionary Value

### What it is
Replaces the value for a specified key in a dictionary.

### How it works
<!-- MANUAL: how_it_works -->
This block updates the value for an existing key in a dictionary. The old value is replaced with the new one, and the updated dictionary is output. The block also returns the old value that was replaced.

This is useful for updating specific fields in a data object while preserving all other fields, or for tracking what value was changed during an update operation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to modify. | Dict[str, Any] | Yes |
| key | Key to replace the value for. | str \| int | Yes |
| value | The new value for the given key. | Value | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_dictionary | The dictionary after replacement. | Dict[str, Any] |
| old_value | The value that was replaced. | Old Value |

### Possible use case
<!-- MANUAL: use_case -->
**Status Updates**: Change the status field in a record from "pending" to "completed" while preserving all other data.

**Configuration Changes**: Update a single setting in a configuration object without rebuilding the entire config.

**Field Transformations**: Replace a raw value with a processed or formatted version while tracking the original.
<!-- END MANUAL -->

---

## Replace List Item

### What it is
Replaces an item at the specified index.

### How it works
<!-- MANUAL: how_it_works -->
This block replaces an item at a specific position in a list with a new value. It uses zero-based indexing and supports negative indices for accessing positions from the end. The old item that was replaced is also returned.

This is useful for updating specific elements in an ordered list without rebuilding the entire list or losing other elements.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list | The list to modify. | List[Any] | Yes |
| index | Index of the item to replace (supports negative indices). | int | Yes |
| value | The new value for the given index. | Value | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_list | The list after replacement. | List[Any] |
| old_item | The item that was replaced. | Old Item |

### Possible use case
<!-- MANUAL: use_case -->
**List Updates**: Replace an outdated entry in a list with updated information while keeping other entries intact.

**Correction Workflows**: Fix a specific item in a results list after validation identifies an error at a known position.

**Value Swapping**: Replace placeholder values in a list with computed or fetched actual values at known positions.
<!-- END MANUAL -->

---

## Reverse List Order

### What it is
Reverses the order of elements in a list

### How it works
<!-- MANUAL: how_it_works -->
This block takes a list and returns a new list with all elements in reverse order. The first element becomes the last, and the last element becomes the first. The original list is not modified.

This is useful for changing the processing order of items or displaying lists in a different order than they were collected.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input_list | The list to reverse | List[Any] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| reversed_list | The list in reversed order | List[Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Chronological Reversal**: Display the most recent items first when data was collected oldest-to-newest.

**Processing Order Change**: Process a stack of items last-in-first-out by reversing a first-in-first-out list.

**Display Formatting**: Reverse leaderboard rankings to show from lowest to highest or vice versa.
<!-- END MANUAL -->

---

## Search Memory

### What it is
Search memories in Mem0 by user

### How it works
<!-- MANUAL: how_it_works -->
This block searches through stored memories using a natural language query. It uses semantic search to find memories that are relevant to your query, not just exact matches. Results can be filtered by categories or metadata.

The search is performed against the Mem0 memory store and returns memories ranked by relevance to your query. This enables intelligent recall of past information based on meaning rather than keywords.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query | str | Yes |
| trigger | An unused field that is used to (re-)trigger the block when you have no other inputs | bool | No |
| categories_filter | Categories to filter by | List[str] | No |
| metadata_filter | Optional metadata filters to apply | Dict[str, Any] | No |
| limit_memory_to_run | Limit the memory to the run | bool | No |
| limit_memory_to_agent | Limit the memory to the agent | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| memories | List of matching memories | Memories |

### Possible use case
<!-- MANUAL: use_case -->
**Contextual Recall**: Search for memories related to a user's current question to provide informed, contextual responses.

**Knowledge Retrieval**: Find previously stored facts or insights that are relevant to a new task or decision.

**Conversation History**: Search past interactions to recall what was discussed about a specific topic or person.
<!-- END MANUAL -->

---

## Store Value

### What it is
A basic block that stores and forwards a value throughout workflows, allowing it to be reused without changes across multiple blocks.

### How it works
<!-- MANUAL: how_it_works -->
It accepts an input value and optionally a data value. If a data value is provided, it is used as the output. Otherwise, the input value is used as the output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input | Trigger the block to produce the output. The value is only used when `data` is None. | Input | Yes |
| data | The constant data to be retained in the block. This value is passed as `output`. | Data | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| output | The stored data retained in the block. | Output |

### Possible use case
<!-- MANUAL: use_case -->
Storing a user's name at the beginning of a workflow to use it in multiple subsequent blocks without asking for it again.
<!-- END MANUAL -->

---

## Universal Type Converter

### What it is
This block is used to convert a value to a universal type.

### How it works
<!-- MANUAL: how_it_works -->
This block converts values between common data types: string, number, boolean, list, and dictionary. It handles type coercion intelligently—for example, converting the string "true" to boolean true, or the string "42" to the number 42.

This is useful when data from different sources needs to be in a consistent type for processing, comparison, or API requirements.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| value | The value to convert to a universal type. | Value | Yes |
| type | The type to convert the value to. | "string" \| "number" \| "boolean" \| "list" \| "dictionary" | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| value | The converted value. | Value |

### Possible use case
<!-- MANUAL: use_case -->
**API Compatibility**: Convert string inputs to numbers or booleans as required by specific API parameters.

**User Input Processing**: Transform user-entered text values into appropriate types for calculations or logic.

**Data Normalization**: Standardize mixed-type data from various sources into consistent types for processing.
<!-- END MANUAL -->

---

## XML Parser

### What it is
Parses XML using gravitasml to tokenize and coverts it to dict

### How it works
<!-- MANUAL: how_it_works -->
This block parses XML content and converts it into a dictionary structure that can be easily navigated and processed in workflows. It uses the gravitasml library to tokenize the XML and produces a nested dictionary matching the XML hierarchy.

This makes XML data accessible using standard dictionary operations, allowing you to extract values, iterate over elements, and process XML-based API responses or data files.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input_xml | input xml to be parsed | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error in parsing | str |
| parsed_xml | output parsed xml to dict | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**API Response Processing**: Parse XML responses from SOAP APIs or legacy systems to extract the data you need.

**Configuration File Reading**: Read XML configuration files and convert them to dictionaries for easy access.

**Data Import**: Transform XML data exports from other systems into a format suitable for your workflow processing.
<!-- END MANUAL -->

---
