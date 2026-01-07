# Create Dictionary

### What it is
Creates a dictionary with the specified key-value pairs.

### What it does
Creates a dictionary with the specified key-value pairs. Use this when you know all the values you want to add upfront.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| values | Key-value pairs to create the dictionary with | Dict[str, True] | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if dictionary creation failed | str |
| dictionary | The created dictionary containing the specified key-value pairs | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Create List

### What it is
Creates a list with the specified values.

### What it does
Creates a list with the specified values. Use this when you know all the values you want to add upfront. This block can also yield the list in batches based on a maximum size or token limit.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## File Read

### What it is
Reads a file and returns its content as a string, with optional chunking by delimiter and size limits.

### What it does
Reads a file and returns its content as a string, with optional chunking by delimiter and size limits

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Persist Information

### What it is
Persist key-value information for the current user.

### What it does
Persist key-value information for the current user

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| key | Key to store the information under | str | Yes |
| value | Value to store | Value | Yes |
| scope | Scope of persistence: within_agent (shared across all runs of this agent) or across_agents (shared across all agents for this user) | "within_agent" | "across_agents" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| value | Value that was stored | Value |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Read Spreadsheet

### What it is
Reads CSV and Excel files and outputs the data as a list of dictionaries and individual rows.

### What it does
Reads CSV and Excel files and outputs the data as a list of dictionaries and individual rows. Excel files are automatically converted to CSV format.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Retrieve Information

### What it is
Retrieve key-value information for the current user.

### What it does
Retrieve key-value information for the current user

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| key | Key to retrieve the information for | str | Yes |
| scope | Scope of persistence: within_agent (shared across all runs of this agent) or across_agents (shared across all agents for this user) | "within_agent" | "across_agents" | No |
| default_value | Default value to return if key is not found | Default Value | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| value | Retrieved value or default value | Value |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Screenshot Web Page

### What it is
Takes a screenshot of a specified website using ScreenshotOne API.

### What it does
Takes a screenshot of a specified website using ScreenshotOne API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | URL of the website to screenshot | str | Yes |
| viewport_width | Width of the viewport in pixels | int | No |
| viewport_height | Height of the viewport in pixels | int | No |
| full_page | Whether to capture the full page length | bool | No |
| format | Output format (png, jpeg, webp) | "png" | "jpeg" | "webp" | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
