# Add Memory

### What it is
Add new memories to Mem0 with user segmentation.

### What it does
Add new memories to Mem0 with user segmentation

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| content | Content to add - either a string or list of message objects as output from an AI block | Content | No |
| metadata | Optional metadata for the memory | Dict[str, True] | No |
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
Adds a new key-value pair to a dictionary.

### What it does
Adds a new key-value pair to a dictionary. If no dictionary is provided, a new one is created.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to add the entry to. If not provided, a new dictionary will be created. | Dict[str, True] | No |
| key | The key for the new entry. | str | No |
| value | The value for the new entry. | Value | No |
| entries | The entries to add to the dictionary. This is the batch version of the `key` and `value` fields. | Dict[str, True] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_dictionary | The dictionary with the new entry added. | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Add To List

### What it is
Adds a new entry to a list.

### What it does
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

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |

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

### What it does
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
| placeholder_values | Possible values for the dropdown. | List[Any] | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |

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

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |
| base_64 | Whether produce an output in base64 format (not recommended, you can pass the string path just fine accross blocks). | bool | No |

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
Block for selecting a file from Google Drive.

### What it does
Block for selecting a file from Google Drive.

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
| secret | Whether the input should be treated as a secret. | bool | No |
| allowed_views | Which views to show in the file picker (DOCS, SPREADSHEETS, PRESENTATIONS, etc.). | List["DOCS" | "DOCUMENTS" | "SPREADSHEETS"] | No |
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
Base block for user inputs.

### What it does
Base block for user inputs.

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

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |

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

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |

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
Stores the output of the graph for users to see.

### What it does
Stores the output of the graph for users to see.

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

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |

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

### What it does
Block for table data input with customizable headers.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| name | The name of the input. | str | Yes |
| value | The table data as a list of dictionaries. | List[Dict[str, True]] | No |
| title | The title of the input. | str | No |
| description | The description of the input. | str | No |
| advanced | Whether to show the input in the advanced section, if the field is not required. | bool | No |
| secret | Whether the input should be treated as a secret. | bool | No |
| column_headers | Column headers for the table. | List[str] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| result | The table data as a list of dictionaries with headers as keys. | List[Dict[str, True]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Agent Time Input

### What it is
Block for time input.

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |

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

### What it does
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
| secret | Whether the input should be treated as a secret. | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| result | Boolean toggle result. | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Dictionary Is Empty

### What it is
Checks if a dictionary is empty.

### What it does
Checks if a dictionary is empty.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to check. | Dict[str, True] | Yes |

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
Stores the input file in the temporary directory.

### What it does
Stores the input file in the temporary directory.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Find In Dictionary

### What it is
Lookup the given key in the input dictionary/object/list and return the value.

### What it does
Lookup the given key in the input dictionary/object/list and return the value.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| input | Dictionary to lookup from | Input | Yes |
| key | Key to lookup in the dictionary | str | int | Yes |

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

### What it does
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

## Get All Memories

### What it is
Retrieve all memories from Mem0 with optional conversation filtering.

### What it does
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
| metadata_filter | Optional metadata filters to apply | Dict[str, True] | No |
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
Retrieve the latest memory from Mem0 with optional key filtering.

### What it does
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
| metadata_filter | Optional metadata filters to apply | Dict[str, True] | No |
| limit_memory_to_run | Limit the memory to the run | bool | No |
| limit_memory_to_agent | Limit the memory to the agent | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| memory | Latest memory if found | Dict[str, True] |
| found | Whether a memory was found | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get List Item

### What it is
Returns the element at the given index.

### What it does
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

### What it does
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
Pause execution and wait for human approval or modification of data.

### What it does
Pause execution and wait for human approval or modification of data

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Installation

### What it is
Given a code string, this block allows the verification and installation of a block code into the system.

### What it does
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

## List Is Empty

### What it is
Checks if a list is empty.

### What it does
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
This block is used to display a sticky note with the given text.

### What it does
This block is used to display a sticky note with the given text.

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
Print the given text to the console, this is used for a debugging purpose.

### What it does
Print the given text to the console, this is used for a debugging purpose.

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

### What it does
Removes a key-value pair from a dictionary.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to modify. | Dict[str, True] | Yes |
| key | Key to remove from the dictionary. | str | int | Yes |
| return_value | Whether to return the removed value. | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_dictionary | The dictionary after removal. | Dict[str, True] |
| removed_value | The removed value if requested. | Removed Value |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Remove From List

### What it is
Removes an item from a list by value or index.

### What it does
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

### What it does
Replaces the value for a specified key in a dictionary.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dictionary | The dictionary to modify. | Dict[str, True] | Yes |
| key | Key to replace the value for. | str | int | Yes |
| value | The new value for the given key. | Value | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| updated_dictionary | The dictionary after replacement. | Dict[str, True] |
| old_value | The value that was replaced. | Old Value |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Replace List Item

### What it is
Replaces an item at the specified index.

### What it does
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
Reverses the order of elements in a list.

### What it does
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
Search memories in Mem0 by user.

### What it does
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
| metadata_filter | Optional metadata filters to apply | Dict[str, True] | No |
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
This block forwards an input value as output, allowing reuse without change.

### What it does
This block forwards an input value as output, allowing reuse without change.

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

### What it does
This block is used to convert a value to a universal type.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| value | The value to convert to a universal type. | Value | Yes |
| type | The type to convert the value to. | "string" | "number" | "boolean" | Yes |

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
Parses XML using gravitasml to tokenize and coverts it to dict.

### What it does
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
| parsed_xml | output parsed xml to dict | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
