# Google Sheets Add Column

### What it is
Add a new column with a header.

### What it does
Add a new column with a header. Can add at the end or insert at a specific position.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| header | Header name for the new column | str | Yes |
| position | Where to add: 'end' for last column, or column letter (e.g., 'C') to insert before | str | No |
| default_value | Default value to fill in all data rows (optional). Requires existing data rows. | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the operation | Dict[str, True] |
| column_letter | Letter of the new column (e.g., 'D') | str |
| column_index | 0-based index of the new column | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Add Dropdown

### What it is
Add a dropdown list (data validation) to cells.

### What it does
Add a dropdown list (data validation) to cells. Useful for enforcing valid inputs.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| range | Cell range to add dropdown to (e.g., 'B2:B100') | str | Yes |
| options | List of dropdown options | List[str] | Yes |
| strict | Reject input not in the list | bool | No |
| show_dropdown | Show dropdown arrow in cells | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the operation | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Add Note

### What it is
Add a note to a cell in a Google Sheet.

### What it does
Add a note to a cell in a Google Sheet

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to add note to | Spreadsheet | No |
| cell | Cell to add note to (e.g., A1, B2) | str | Yes |
| note | Note text to add | str | Yes |
| sheet_name | Name of the sheet. Defaults to first sheet. | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the operation | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Append Row

### What it is
Append or Add a single row to the end of a Google Sheet.

### What it does
Append or Add a single row to the end of a Google Sheet. The row is added after the last row with data.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| row | Row values to append (e.g., ['Alice', 'alice@example.com', '25']) | List[str] | Yes |
| sheet_name | Sheet to append to (optional, defaults to first sheet) | str | No |
| value_input_option | How values are interpreted. USER_ENTERED: parsed like typed input (e.g., '=SUM(A1:A5)' becomes a formula, '1/2/2024' becomes a date). RAW: stored as-is without parsing. | "RAW" | "USER_ENTERED" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Append API response | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining to other blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Batch Operations

### What it is
This block performs multiple operations on a Google Sheets spreadsheet in a single batch request.

### What it does
This block performs multiple operations on a Google Sheets spreadsheet in a single batch request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| operations | List of operations to perform | List[BatchOperation] | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the batch operations | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Clear

### What it is
This block clears data from a specified range in a Google Sheets spreadsheet.

### What it does
This block clears data from a specified range in a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| range | The A1 notation of the range to clear | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the clear operation | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Copy To Spreadsheet

### What it is
Copy a sheet from one spreadsheet to another.

### What it does
Copy a sheet from one spreadsheet to another.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| source_spreadsheet | Select the source spreadsheet | Source Spreadsheet | No |
| source_sheet_name | Sheet to copy (optional, defaults to first sheet) | str | No |
| destination_spreadsheet_id | ID of the destination spreadsheet | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the copy operation | Dict[str, True] |
| new_sheet_id | ID of the new sheet in the destination | int |
| new_sheet_name | Name of the new sheet | str |
| spreadsheet | The source spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Create Named Range

### What it is
Create a named range to reference cells by name instead of A1 notation.

### What it does
Create a named range to reference cells by name instead of A1 notation.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| name | Name for the range (e.g., 'SalesData', 'CustomerList') | str | Yes |
| range | Cell range in A1 notation (e.g., 'A1:D10', 'B2:B100') | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the operation | Dict[str, True] |
| named_range_id | ID of the created named range | str |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Create Spreadsheet

### What it is
This block creates a new Google Sheets spreadsheet with specified sheets.

### What it does
This block creates a new Google Sheets spreadsheet with specified sheets.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | The title of the new spreadsheet | str | Yes |
| sheet_names | List of sheet names to create (optional, defaults to single 'Sheet1') | List[str] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result containing spreadsheet ID and URL | Dict[str, True] |
| spreadsheet | The created spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |
| spreadsheet_id | The ID of the created spreadsheet | str |
| spreadsheet_url | The URL of the created spreadsheet | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Delete Column

### What it is
Delete a column by header name or column letter.

### What it does
Delete a column by header name or column letter.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| column | Column to delete (header name or column letter like 'A', 'B') | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the delete operation | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Delete Rows

### What it is
Delete specific rows from a Google Sheet by their row indices.

### What it does
Delete specific rows from a Google Sheet by their row indices. Works well with FilterRowsBlock output.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| row_indices | 1-based row indices to delete (e.g., [2, 5, 7]) | List[int] | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the delete operation | Dict[str, True] |
| deleted_count | Number of rows deleted | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Export Csv

### What it is
Export a Google Sheet as CSV data.

### What it does
Export a Google Sheet as CSV data

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to export from | Spreadsheet | No |
| sheet_name | Name of the sheet to export. Defaults to first sheet. | str | No |
| include_headers | Include the first row (headers) in the CSV output | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if export failed | str |
| csv_data | The sheet data as CSV string | str |
| row_count | Number of rows exported | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Filter Rows

### What it is
Filter rows in a Google Sheet based on a column condition.

### What it does
Filter rows in a Google Sheet based on a column condition. Returns matching rows and their indices.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| filter_column | Column to filter on (header name or column letter like 'A', 'B') | str | Yes |
| filter_value | Value to filter by (not used for is_empty/is_not_empty operators) | str | No |
| operator | Filter comparison operator | "equals" | "not_equals" | "contains" | No |
| match_case | Whether to match case in comparisons | bool | No |
| include_header | Include header row in output | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| rows | Filtered rows (including header if requested) | List[List[str]] |
| row_indices | Original 1-based row indices of matching rows (useful for deletion) | List[int] |
| count | Number of matching rows (excluding header) | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Find

### What it is
Find text in a Google Sheets spreadsheet.

### What it does
Find text in a Google Sheets spreadsheet. Returns locations and count of occurrences. Can find all occurrences or just the first one.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| find_text | The text to find | str | Yes |
| sheet_id | The ID of the specific sheet to search (optional, searches all sheets if not provided) | int | No |
| match_case | Whether to match case | bool | No |
| match_entire_cell | Whether to match entire cell | bool | No |
| find_all | Whether to find all occurrences (true) or just the first one (false) | bool | No |
| range | The A1 notation range to search in (optional, searches entire sheet if not provided) | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the find operation including locations and count | Dict[str, True] |
| locations | List of cell locations where the text was found | List[Dict[str, True]] |
| count | Number of occurrences found | int |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Find Replace

### What it is
This block finds and replaces text in a Google Sheets spreadsheet.

### What it does
This block finds and replaces text in a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| find_text | The text to find | str | Yes |
| replace_text | The text to replace with | str | Yes |
| sheet_id | The ID of the specific sheet to search (optional, searches all sheets if not provided) | int | No |
| match_case | Whether to match case | bool | No |
| match_entire_cell | Whether to match entire cell | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the find/replace operation including number of replacements | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Format

### What it is
Format a range in a Google Sheet (sheet optional).

### What it does
Format a range in a Google Sheet (sheet optional)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| range | A1 notation – sheet optional | str | Yes |
| background_color | - | Dict[str, True] | No |
| text_color | - | Dict[str, True] | No |
| bold | - | bool | No |
| italic | - | bool | No |
| font_size | - | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | API response or success flag | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Get Column

### What it is
Extract all values from a specific column.

### What it does
Extract all values from a specific column. Useful for getting a list of emails, IDs, or any single field.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| column | Column to extract (header name or column letter like 'A', 'B') | str | Yes |
| include_header | Include the header in output | bool | No |
| skip_empty | Skip empty cells | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| values | List of values from the column | List[str] |
| count | Number of values (excluding header if not included) | int |
| column_index | 0-based column index | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Get Notes

### What it is
Get notes from cells in a Google Sheet.

### What it does
Get notes from cells in a Google Sheet

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to get notes from | Spreadsheet | No |
| range | Range to get notes from (e.g., A1:B10) | str | No |
| sheet_name | Name of the sheet. Defaults to first sheet. | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| notes | List of notes with cell and text | List[Dict[str, True]] |
| count | Number of notes found | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Get Row

### What it is
Get a specific row by its index.

### What it does
Get a specific row by its index. Returns both list and dict formats.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| row_index | 1-based row index to retrieve | int | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| row | The row values as a list | List[str] |
| row_dict | The row as a dictionary (header: value) | Dict[str, str] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Get Row Count

### What it is
Get row count and dimensions of a Google Sheet.

### What it does
Get row count and dimensions of a Google Sheet. Useful for knowing where data ends.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| include_header | Include header row in count | bool | No |
| count_empty | Count rows with only empty cells | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| total_rows | Total number of rows | int |
| data_rows | Number of data rows (excluding header) | int |
| last_row | 1-based index of the last row with data | int |
| column_count | Number of columns | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Get Unique Values

### What it is
Get unique values from a column.

### What it does
Get unique values from a column. Useful for building dropdown options or finding distinct categories.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| column | Column to get unique values from (header name or column letter) | str | Yes |
| include_count | Include count of each unique value | bool | No |
| sort_by_count | Sort results by count (most frequent first) | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| values | List of unique values | List[str] |
| counts | Count of each unique value (if include_count is True) | Dict[str, int] |
| total_unique | Total number of unique values | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Import Csv

### What it is
Import CSV data into a Google Sheet.

### What it does
Import CSV data into a Google Sheet

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to import into | Spreadsheet | No |
| csv_data | CSV data to import | str | Yes |
| sheet_name | Name of the sheet. Defaults to first sheet. | str | No |
| start_cell | Cell to start importing at (e.g., A1, B2) | str | No |
| clear_existing | Clear existing data before importing | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if import failed | str |
| result | Import result | Dict[str, True] |
| rows_imported | Number of rows imported | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Insert Row

### What it is
Insert a single row at a specific position.

### What it does
Insert a single row at a specific position. Existing rows shift down.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| row | Row values to insert (e.g., ['Alice', 'alice@example.com', '25']) | List[str] | Yes |
| row_index | 1-based row index where to insert (existing rows shift down) | int | Yes |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| value_input_option | How values are interpreted. USER_ENTERED: parsed like typed input (e.g., '=SUM(A1:A5)' becomes a formula, '1/2/2024' becomes a date). RAW: stored as-is without parsing. | "RAW" | "USER_ENTERED" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the insert operation | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets List Named Ranges

### What it is
List all named ranges in a spreadsheet.

### What it does
List all named ranges in a spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| named_ranges | List of named ranges with name, id, and range info | List[Dict[str, True]] |
| count | Number of named ranges | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Lookup Row

### What it is
Look up a row by finding a value in a specific column.

### What it does
Look up a row by finding a value in a specific column. Returns the first matching row and optionally specific columns.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| lookup_column | Column to search in (header name or column letter) | str | Yes |
| lookup_value | Value to search for | str | Yes |
| return_columns | Columns to return (header names or letters). Empty = all columns. | List[str] | No |
| match_case | Whether to match case | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| row | The matching row (all or selected columns) | List[str] |
| row_dict | The matching row as a dictionary (header: value) | Dict[str, str] |
| row_index | 1-based row index of the match | int |
| found | Whether a match was found | bool |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Manage Sheet

### What it is
Create, delete, or copy sheets (sheet optional).

### What it does
Create, delete, or copy sheets (sheet optional)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| operation | Operation to perform | "create" | "delete" | "copy" | Yes |
| sheet_name | Target sheet name (defaults to first sheet for delete) | str | No |
| source_sheet_id | Source sheet ID for copy | int | No |
| destination_sheet_name | New sheet name for copy | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Operation result | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Metadata

### What it is
This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties.

### What it does
This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The metadata of the spreadsheet including sheets info | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Protect Range

### What it is
Protect a cell range or entire sheet from editing.

### What it does
Protect a cell range or entire sheet from editing.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| range | Cell range to protect (e.g., 'A1:D10'). Leave empty to protect entire sheet. | str | No |
| description | Description for the protected range | str | No |
| warning_only | Show warning but allow editing (vs blocking completely) | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the operation | Dict[str, True] |
| protection_id | ID of the protection | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Read

### What it is
This block reads data from a Google Sheets spreadsheet.

### What it does
This block reads data from a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to Google Sheets using provided credentials, then fetches data from the specified spreadsheet and range.
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| range | The A1 notation of the range to read | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The data read from the spreadsheet | List[List[str]] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
A marketing team could use this block to automatically retrieve the latest campaign performance data from a shared Google Sheets document for analysis and reporting.
<!-- END MANUAL -->

---

## Google Sheets Remove Duplicates

### What it is
Remove duplicate rows based on specified columns.

### What it does
Remove duplicate rows based on specified columns. Keeps either the first or last occurrence.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| columns | Columns to check for duplicates (header names or letters). Empty = all columns. | List[str] | No |
| keep | Which duplicate to keep: 'first' or 'last' | str | No |
| match_case | Whether to match case when comparing | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the operation | Dict[str, True] |
| removed_count | Number of duplicate rows removed | int |
| remaining_rows | Number of rows remaining | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Set Public Access

### What it is
Make a Google Spreadsheet public or private.

### What it does
Make a Google Spreadsheet public or private

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to modify access for | Spreadsheet | No |
| public | True to make public, False to make private | bool | No |
| role | Permission role for public access | "reader" | "commenter" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the operation | Dict[str, True] |
| share_link | Link to the spreadsheet | str |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Share Spreadsheet

### What it is
Share a Google Spreadsheet with users or get shareable link.

### What it does
Share a Google Spreadsheet with users or get shareable link

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to share | Spreadsheet | No |
| email | Email address to share with. Leave empty for link sharing. | str | No |
| role | Permission role for the user | "reader" | "writer" | "commenter" | No |
| send_notification | Send notification email to the user | bool | No |
| message | Optional message to include in notification email | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if share failed | str |
| result | Result of the share operation | Dict[str, True] |
| share_link | Link to the spreadsheet | str |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Sort

### What it is
Sort a Google Sheet by one or two columns.

### What it does
Sort a Google Sheet by one or two columns. The sheet is sorted in-place.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| sort_column | Primary column to sort by (header name or column letter) | str | Yes |
| sort_order | Sort order for primary column | "ascending" | "descending" | No |
| secondary_column | Secondary column to sort by (optional) | str | No |
| secondary_order | Sort order for secondary column | "ascending" | "descending" | No |
| has_header | Whether the data has a header row (header won't be sorted) | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the sort operation | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Update Cell

### What it is
Update a single cell in a Google Sheets spreadsheet.

### What it does
Update a single cell in a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| cell | Cell address in A1 notation (e.g., 'A1', 'Sheet1!B2') | str | Yes |
| value | Value to write to the cell | str | Yes |
| value_input_option | How input data should be interpreted | "RAW" | "USER_ENTERED" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the update operation | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Update Row

### What it is
Update a specific row by its index.

### What it does
Update a specific row by its index. Can use list or dict format for values.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| row_index | 1-based row index to update | int | Yes |
| values | New values for the row (in column order) | List[str] | No |
| dict_values | Values as dict with column headers as keys (alternative to values) | Dict[str, str] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the update operation | Dict[str, True] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Google Sheets Write

### What it is
This block writes data to a Google Sheets spreadsheet.

### What it does
This block writes data to a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
The block authenticates with Google Sheets using provided credentials, then updates the specified spreadsheet range with the given data.
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| range | The A1 notation of the range to write | str | Yes |
| values | The data to write to the spreadsheet | List[List[str]] | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the write operation | Dict[str, True] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
An automated inventory system could use this block to update stock levels in a Google Sheets spreadsheet whenever products are sold or restocked, ensuring real-time inventory tracking.
<!-- END MANUAL -->

---
