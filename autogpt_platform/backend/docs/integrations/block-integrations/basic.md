# Basic
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Add Memory

### What it is
Add new memories to Mem0 with user segmentation

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Add To Dictionary

### What it is
Adds a new key-value pair to a dictionary. If no dictionary is provided, a new one is created.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Add To List

### What it is
Adds a new entry to a list. The entry can be of any type. If no list is provided, a new one is created.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Date Input

### What it is
Block for date input.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Date input (YYYY-MM-DD). | str (date) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Date result. | str (date) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Dropdown Input

### What it is
Block for dropdown text selection.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Text selected from a dropdown. | str | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| options | If provided, renders the input as a dropdown selector restricted to these values. Leave empty for free-text input. | List[Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Selected dropdown value. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent File Input

### What it is
Block for file upload input (string path for example).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Path or reference to an uploaded file. | str (file) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| base_64 | Whether to produce output in base64 format (not recommended; you can pass the file reference across blocks). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | File reference/path result. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Google Drive File Input

### What it is
Agent-level input for a Google Drive file. REQUIRED for any agent that reads or writes a Drive file (Sheets, Docs, Slides, or generic Drive) — the picker is the only source of the _credentials_id needed at runtime, so consuming blocks cannot receive a hardcoded ID. Set allowed_views to match the consumer: ["SPREADSHEETS"] for Sheets, ["DOCUMENTS"] for Docs, ["PRESENTATIONS"] for Slides (leave default for generic Drive). Wire `result` to the consumer block's Drive field and leave that field unset in the consumer's input_default. Example link to a Google Sheets block: {"source_name": "result", "sink_name": "spreadsheet"} (use "document" for Docs, "presentation" for Slides). Use one input block per distinct file; multiple consumers of the same file share it.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The selected Google Drive file. | GoogleDriveFile | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| allowed_views | Which views to show in the file picker (DOCS, SPREADSHEETS, PRESENTATIONS, etc.). | List["DOCS" \| "DOCUMENTS" \| "SPREADSHEETS" \| "PRESENTATIONS" \| "DOCS_IMAGES" \| "FOLDERS"] | No |
| allow_folder_selection | Whether to allow selecting folders. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | The selected Google Drive file with ID, name, URL, and other metadata. | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Input

### What it is
A block that accepts and processes user input values within a workflow, supporting various input types and validation.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The value to be passed as input. | Value | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | The value passed as input. | Result |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Long Text Input

### What it is
Block for long text input (multi-line).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Long text input (potentially multi-line). | str (long-text) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Long text result. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Number Input

### What it is
Block for number input.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Number input. | int | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Number result. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Output

### What it is
A block that records and formats workflow results for display to users, with optional Jinja2 template formatting support.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| output | The value recorded as output. | Output |
| name | The name of the value recorded as output. | Name |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Short Text Input

### What it is
Block for short text input (single-line).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Short text input. | str (short-text) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Short text result. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Table Input

### What it is
Block for table data input with customizable headers.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The table data as a list of dictionaries. | List[Dict[str, Any]] | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| column_headers | Column headers for the table. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | The table data as a list of dictionaries with headers as keys. | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Time Input

### What it is
Block for time input.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Time input (HH:MM:SS). | str (time) | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Time result. | str (time) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Toggle Input

### What it is
Block for boolean toggle input.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | Boolean toggle input. | bool | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| result | Boolean toggle result. | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Block Installation

### What it is
Given a code string, this block allows the verification and installation of a block code into the system.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Concatenate Lists

### What it is
Concatenates multiple lists into a single list. All elements from all input lists are combined in order. Supports optional deduplication and None removal.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| lists | A list of lists to concatenate together. All lists will be combined in order into a single list. | List[List[Any]] | Yes |
| deduplicate | If True, remove duplicate elements from the concatenated result while preserving order. | bool | No |
| remove_none | If True, remove None values from the concatenated result. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if concatenation failed due to invalid input types. | str |
| concatenated_list | The concatenated list containing all elements from all input lists in order. | List[Any] |
| length | The total number of elements in the concatenated list. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Dictionary Is Empty

### What it is
Checks if a dictionary is empty.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## File Store

### What it is
Downloads and stores a file from a URL, data URI, or local path. Use this to fetch images, documents, or other files for processing. In CoPilot: saves to workspace (use list_workspace_files to see it). In graphs: outputs a data URI to pass to other blocks.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_in | The file to download and store. Can be a URL (https://...), data URI, or local path. | str (file) | Yes |
| base_64 | Whether to produce output in base64 format (not recommended, you can pass the file reference across blocks). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| file_out | Reference to the stored file. In CoPilot: workspace:// URI (visible in list_workspace_files). In graphs: data URI for passing to other blocks. | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Find In Dictionary

### What it is
A block that looks up a value in a dictionary, list, or object by key or index and returns the corresponding value.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Find In List

### What it is
Finds the index of the value in the list.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Flatten List

### What it is
Flattens a nested list structure into a single flat list. Supports configurable maximum flattening depth.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| nested_list | A potentially nested list to flatten into a single-level list. | List[Any] | Yes |
| max_depth | Maximum depth to flatten. -1 means flatten completely. 1 means flatten only one level. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if flattening failed. | str |
| flattened_list | The flattened list with all nested elements extracted. | List[Any] |
| length | The number of elements in the flattened list. | int |
| original_depth | The maximum nesting depth of the original input list. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get All Memories

### What it is
Retrieve all memories from Mem0 with optional conversation filtering

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Latest Memory

### What it is
Retrieve the latest memory from Mem0 with optional key filtering

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get List Item

### What it is
Returns the element at the given index.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Weather Information

### What it is
Retrieves weather information for a specified location using OpenWeatherMap API.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Human In The Loop

### What it is
Pause execution for human review. Data flows through approved_data or rejected_data output based on the reviewer's decision. Outputs contain the actual data, not status strings.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| data | The data to be reviewed by a human user. This exact data will be passed through to either approved_data or rejected_data output based on the reviewer's decision. | Data | Yes |
| name | A descriptive name for what this data represents. This helps the reviewer understand what they are reviewing. | str | Yes |
| editable | Whether the human reviewer can edit the data before approving or rejecting it | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| approved_data | Outputs the input data when the reviewer APPROVES it. The value is the actual data itself (not a status string like 'APPROVED'). If the reviewer edited the data, this contains the modified version. Connect downstream blocks here for the 'approved' workflow path. | Approved Data |
| rejected_data | Outputs the input data when the reviewer REJECTS it. The value is the actual data itself (not a status string like 'REJECTED'). If the reviewer edited the data, this contains the modified version. Connect downstream blocks here for the 'rejected' workflow path. | Rejected Data |
| review_message | Optional message provided by the reviewer explaining their decision. Only outputs when the reviewer provides a message; this pin does not fire if no message was given. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Interleave Lists

### What it is
Interleaves elements from multiple lists in round-robin fashion, alternating between sources.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| lists | A list of lists to interleave. Elements will be taken in round-robin order. | List[List[Any]] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if interleaving failed. | str |
| interleaved_list | The interleaved list with elements alternating from each input list. | List[Any] |
| length | The total number of elements in the interleaved list. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## List Difference

### What it is
Computes the difference between two lists. Returns elements in the first list not found in the second, or symmetric difference.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_a | The primary list to check elements from. | List[Any] | Yes |
| list_b | The list to subtract. Elements found here will be removed from list_a. | List[Any] | Yes |
| symmetric | If True, compute symmetric difference (elements in either list but not both). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed. | str |
| difference | Elements from list_a not found in list_b (or symmetric difference if enabled). | List[Any] |
| length | The number of elements in the difference result. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## List Intersection

### What it is
Computes the intersection of two lists, returning only elements present in both.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_a | The first list to intersect. | List[Any] | Yes |
| list_b | The second list to intersect. | List[Any] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed. | str |
| intersection | Elements present in both list_a and list_b. | List[Any] |
| length | The number of elements in the intersection. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## List Is Empty

### What it is
Checks if a list is empty.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Note

### What it is
A visual annotation block that displays a sticky note in the workflow editor for documentation and organization purposes.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Print To Console

### What it is
A debugging block that outputs text to the console for monitoring and troubleshooting workflow execution.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Remove From Dictionary

### What it is
Removes a key-value pair from a dictionary.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Remove From List

### What it is
Removes an item from a list by value or index.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Replace Dictionary Value

### What it is
Replaces the value for a specified key in a dictionary.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Replace List Item

### What it is
Replaces an item at the specified index.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Reverse List Order

### What it is
Reverses the order of elements in a list

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Search Memory

### What it is
Search memories in Mem0 by user

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Store Value

### What it is
A basic block that stores and forwards a value throughout workflows, allowing it to be reused without changes across multiple blocks.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Universal Type Converter

### What it is
This block is used to convert a value to a universal type.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## XML Parser

### What it is
Parses XML using gravitasml to tokenize and coverts it to dict

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Zip Lists

### What it is
Zips multiple lists together into a list of grouped elements. Supports padding to longest or truncating to shortest.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| lists | A list of lists to zip together. Corresponding elements will be grouped. | List[List[Any]] | Yes |
| pad_to_longest | If True, pad shorter lists with fill_value to match the longest list. If False, truncate to shortest. | bool | No |
| fill_value | Value to use for padding when pad_to_longest is True. | Fill Value | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if zipping failed. | str |
| zipped_list | The zipped list of grouped elements. | List[List[Any]] |
| length | The number of groups in the zipped result. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
