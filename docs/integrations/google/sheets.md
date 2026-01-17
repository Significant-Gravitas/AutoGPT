# Google Sheets
<!-- MANUAL: file_description -->
Blocks for creating, reading, and editing Google Sheets spreadsheets.
<!-- END MANUAL -->

## Google Sheets Add Column

### What it is
Add a new column with a header. Can add at the end or insert at a specific position.

### How it works
<!-- MANUAL: how_it_works -->
This block adds a new column to a Google Sheet with a specified header name. You can add the column at the end or insert it before a specific column position. If you provide a default value, all existing data rows will be populated with that value.

The block uses the Google Sheets API to perform the insertion, shifting existing columns to the right when inserting in the middle.
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
| result | Result of the operation | Dict[str, Any] |
| column_letter | Letter of the new column (e.g., 'D') | str |
| column_index | 0-based index of the new column | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Schema Evolution**: Add new data fields to existing tracking sheets as requirements change.

**Status Tracking**: Add a "Status" or "Processed" column to mark items as they flow through a workflow.

**Data Enhancement**: Add computed or lookup columns to enrich existing datasets.
<!-- END MANUAL -->

---

## Google Sheets Add Dropdown

### What it is
Add a dropdown list (data validation) to cells. Useful for enforcing valid inputs.

### How it works
<!-- MANUAL: how_it_works -->
This block adds data validation rules to a cell range using the Google Sheets API, creating dropdown menus with predefined options. You can enforce strict validation (reject invalid inputs) or show warnings for non-list values.

The dropdown arrow appears in cells when enabled, providing users with a list of valid choices.
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
| result | Result of the operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Status Tracking**: Add status dropdowns (Pending, In Progress, Complete) to task tracking sheets.

**Data Entry Control**: Restrict input options to valid categories, departments, or product types.

**Survey Forms**: Create structured input fields with predefined response options.
<!-- END MANUAL -->

---

## Google Sheets Add Note

### What it is
Add a note to a cell in a Google Sheet

### How it works
<!-- MANUAL: how_it_works -->
This block adds a text note to a specific cell using the Google Sheets API. Notes appear when hovering over a cell and provide additional context without affecting cell values.

Notes are useful for documentation, explanations, or audit trails that shouldn't modify the underlying data.
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
| result | Result of the operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Audit Trail**: Add notes documenting who modified data and when.

**Data Documentation**: Explain calculation logic or data sources for specific cells.

**Error Flagging**: Mark cells with notes explaining validation issues or anomalies.
<!-- END MANUAL -->

---

## Google Sheets Append Row

### What it is
Append or Add a single row to the end of a Google Sheet. The row is added after the last row with data.

### How it works
<!-- MANUAL: how_it_works -->
This block appends a new row to the end of a Google Sheet after the last row containing data. You provide values as a list that maps to columns in order. The value_input_option controls whether values are parsed (like formulas or dates) or stored as raw text.

This is ideal for continuously adding records to a log or database-style sheet.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| row | Row values to append (e.g., ['Alice', 'alice@example.com', '25']) | List[str] | Yes |
| sheet_name | Sheet to append to (optional, defaults to first sheet) | str | No |
| value_input_option | How values are interpreted. USER_ENTERED: parsed like typed input (e.g., '=SUM(A1:A5)' becomes a formula, '1/2/2024' becomes a date). RAW: stored as-is without parsing. | "RAW" \| "USER_ENTERED" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Append API response | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining to other blocks | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Capture**: Append new leads from forms or webhooks to a sales tracking sheet.

**Event Logging**: Add timestamped entries to an activity log or audit trail.

**Data Collection**: Continuously gather data points from sensors, APIs, or user submissions.
<!-- END MANUAL -->

---

## Google Sheets Batch Operations

### What it is
This block performs multiple operations on a Google Sheets spreadsheet in a single batch request.

### How it works
<!-- MANUAL: how_it_works -->
This block combines multiple spreadsheet operations into a single API call using the Google Sheets API's batch update functionality. This is more efficient than making separate calls for each operation.

Operations execute in order and can include various actions like formatting, data validation, and protection. The batch fails if any operation fails.
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
| result | The result of the batch operations | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Bulk Setup**: Apply multiple formatting rules, validations, and protections when setting up new sheets.

**Performance Optimization**: Reduce API calls by batching multiple updates together.

**Atomic Updates**: Ensure multiple related changes succeed or fail together.
<!-- END MANUAL -->

---

## Google Sheets Clear

### What it is
This block clears data from a specified range in a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
This block removes all values from a specified cell range using the Google Sheets API. The cells remain but their contents are deleted, preserving any formatting.

Use A1 notation (e.g., "A1:D10" or "Sheet1!B2:C5") to specify the range to clear.
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
| result | The result of the clear operation | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Reset**: Clear data areas before importing fresh data from external sources.

**Template Preparation**: Reset input areas of templates before distributing new copies.

**Periodic Cleanup**: Clear temporary or staging data areas as part of scheduled workflows.
<!-- END MANUAL -->

---

## Google Sheets Copy To Spreadsheet

### What it is
Copy a sheet from one spreadsheet to another.

### How it works
<!-- MANUAL: how_it_works -->
This block copies an entire sheet (tab) from one Google Spreadsheet to another using the Google Sheets API. The copied sheet includes all data, formatting, formulas, and structure.

The new sheet is added to the destination spreadsheet with a potentially modified name if a sheet with the same name already exists.
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
| result | Result of the copy operation | Dict[str, Any] |
| new_sheet_id | ID of the new sheet in the destination | int |
| new_sheet_name | Name of the new sheet | str |
| spreadsheet | The source spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Template Distribution**: Copy template sheets to individual project or client spreadsheets.

**Report Consolidation**: Copy data sheets from multiple sources into a master spreadsheet.

**Backup Creation**: Copy important sheets to backup spreadsheets for archival.
<!-- END MANUAL -->

---

## Google Sheets Create Named Range

### What it is
Create a named range to reference cells by name instead of A1 notation.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a named range in a Google Spreadsheet using the Google Sheets API. Named ranges allow you to reference cells by descriptive names instead of A1 notation, making formulas more readable.

Named ranges can be used in formulas across the spreadsheet and make maintenance easier when data locations change.
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
| result | Result of the operation | Dict[str, Any] |
| named_range_id | ID of the created named range | str |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Formula Clarity**: Create named ranges like "SalesData" or "TaxRate" for clearer formulas.

**Dynamic References**: Define named ranges that formulas reference, simplifying updates.

**Data Source Setup**: Create named ranges for areas that serve as data sources for charts or lookups.
<!-- END MANUAL -->

---

## Google Sheets Create Spreadsheet

### What it is
This block creates a new Google Sheets spreadsheet with specified sheets.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a brand new Google Sheets spreadsheet with a specified title and optional sheet names. The spreadsheet is created in the user's Google Drive and immediately accessible via the returned URL.

The spreadsheet output can be chained to other Sheets blocks for immediate data population.
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
| result | The result containing spreadsheet ID and URL | Dict[str, Any] |
| spreadsheet | The created spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |
| spreadsheet_id | The ID of the created spreadsheet | str |
| spreadsheet_url | The URL of the created spreadsheet | str |

### Possible use case
<!-- MANUAL: use_case -->
**Report Generation**: Create new spreadsheets for periodic reports with dedicated sheets for different data sections.

**Project Setup**: Automatically create project tracking spreadsheets with pre-defined sheet structures.

**Data Export**: Create new spreadsheets to export data from other systems for sharing.
<!-- END MANUAL -->

---

## Google Sheets Delete Column

### What it is
Delete a column by header name or column letter.

### How it works
<!-- MANUAL: how_it_works -->
This block removes an entire column from a Google Sheet using the Google Sheets API. You can specify the column by its header name or column letter (A, B, C, etc.).

All data in the column is permanently deleted and subsequent columns shift left to fill the gap.
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
| result | Result of the delete operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Schema Cleanup**: Remove deprecated or unused columns from data sheets.

**Data Reduction**: Delete columns containing sensitive or unnecessary information before sharing.

**Structure Maintenance**: Remove temporary calculation columns after processing.
<!-- END MANUAL -->

---

## Google Sheets Delete Rows

### What it is
Delete specific rows from a Google Sheet by their row indices. Works well with FilterRowsBlock output.

### How it works
<!-- MANUAL: how_it_works -->
This block removes specific rows from a Google Sheet by their 1-based row indices using the Google Sheets API. It handles multiple deletions efficiently by processing from bottom to top to maintain correct indices.

Works seamlessly with the Filter Rows block output to delete rows matching specific criteria.
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
| result | Result of the delete operation | Dict[str, Any] |
| deleted_count | Number of rows deleted | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Cleanup**: Delete rows with invalid or incomplete data identified by filtering.

**Record Removal**: Remove processed or archived records from active sheets.

**Conditional Deletion**: Delete rows matching specific criteria like outdated entries or cancelled items.
<!-- END MANUAL -->

---

## Google Sheets Export Csv

### What it is
Export a Google Sheet as CSV data

### How it works
<!-- MANUAL: how_it_works -->
This block exports data from a Google Sheet as a CSV-formatted string using the Google Sheets API. You can choose whether to include headers and specify which sheet to export.

The CSV data can be used for integration with other systems, file downloads, or data processing pipelines.
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
**System Integration**: Export data as CSV for import into external systems or databases.

**File Distribution**: Generate CSV exports for email attachments or file downloads.

**Data Backup**: Create periodic CSV backups of spreadsheet data.
<!-- END MANUAL -->

---

## Google Sheets Filter Rows

### What it is
Filter rows in a Google Sheet based on a column condition. Returns matching rows and their indices.

### How it works
<!-- MANUAL: how_it_works -->
This block filters rows in a Google Sheet based on conditions applied to a specific column using the Google Sheets API. Supports operators like equals, not_equals, contains, is_empty, and is_not_empty.

Returns matching rows along with their original 1-based row indices, making it easy to chain with delete or update operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| filter_column | Column to filter on (header name or column letter like 'A', 'B') | str | Yes |
| filter_value | Value to filter by (not used for is_empty/is_not_empty operators) | str | No |
| operator | Filter comparison operator | "equals" \| "not_equals" \| "contains" \| "not_contains" \| "greater_than" \| "less_than" \| "greater_than_or_equal" \| "less_than_or_equal" \| "is_empty" \| "is_not_empty" | No |
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
**Conditional Processing**: Filter rows by status to process only pending or active items.

**Data Extraction**: Extract rows matching specific criteria for reports or analysis.

**Cleanup Preparation**: Identify rows to delete based on conditions like empty values or specific statuses.
<!-- END MANUAL -->

---

## Google Sheets Find

### What it is
Find text in a Google Sheets spreadsheet. Returns locations and count of occurrences. Can find all occurrences or just the first one.

### How it works
<!-- MANUAL: how_it_works -->
This block searches for text across a Google Spreadsheet using the Google Sheets API. You can search entire spreadsheets or specific sheets/ranges, with options for case matching and whole-cell matching.

Returns the locations (sheet, row, column) of all matches or just the first one, along with the total count.
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
| result | The result of the find operation including locations and count | Dict[str, Any] |
| locations | List of cell locations where the text was found | List[Dict[str, Any]] |
| count | Number of occurrences found | int |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Validation**: Find specific values to verify data presence or locate errors.

**Position Lookup**: Find where specific text appears to determine update locations.

**Audit Checks**: Search for specific terms across spreadsheets for compliance verification.
<!-- END MANUAL -->

---

## Google Sheets Find Replace

### What it is
This block finds and replaces text in a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
This block performs find-and-replace operations across a Google Spreadsheet using the Google Sheets API. You can target all sheets or specific ones, with options for case matching and whole-cell matching.

Returns the number of replacements made, enabling verification of the operation's scope.
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
| result | The result of the find/replace operation including number of replacements | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Standardization**: Replace variations of terms with standardized values.

**Batch Updates**: Update company names, status values, or codes across entire spreadsheets.

**Error Correction**: Fix systematic typos or outdated values across all sheets.
<!-- END MANUAL -->

---

## Google Sheets Format

### What it is
Format a range in a Google Sheet (sheet optional)

### How it works
<!-- MANUAL: how_it_works -->
This block applies visual formatting to cell ranges in a Google Sheet using the Google Sheets API. Options include background color, text color, bold, italic, and font size.

Formatting enhances readability and can highlight important data or create visual structure in reports.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| range | A1 notation – sheet optional | str | Yes |
| background_color | - | Dict[str, Any] | No |
| text_color | - | Dict[str, Any] | No |
| bold | - | bool | No |
| italic | - | bool | No |
| font_size | - | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | API response or success flag | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Header Styling**: Format header rows with bold text and background colors.

**Conditional Highlighting**: Apply colors to highlight important values or warnings.

**Report Presentation**: Style sheets for professional presentation before sharing.
<!-- END MANUAL -->

---

## Google Sheets Get Column

### What it is
Extract all values from a specific column. Useful for getting a list of emails, IDs, or any single field.

### How it works
<!-- MANUAL: how_it_works -->
This block extracts all values from a specific column in a Google Sheet using the Google Sheets API. You can reference the column by header name or column letter, with options to skip empty cells and include/exclude headers.

Returns values as a list for easy iteration or processing in subsequent blocks.
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
**Email Lists**: Extract email column values for sending notifications or marketing.

**ID Processing**: Get all IDs from a column to process records individually.

**Data Validation**: Extract values to check against allowed lists or external databases.
<!-- END MANUAL -->

---

## Google Sheets Get Notes

### What it is
Get notes from cells in a Google Sheet

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves cell notes from a specified range in a Google Sheet using the Google Sheets API. Notes are the comments that appear when hovering over cells.

Returns a list of notes with their cell locations, useful for extracting documentation or audit information.
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
| notes | List of notes with cell and text | List[Dict[str, Any]] |
| count | Number of notes found | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Audit Extraction**: Retrieve notes documenting changes or approvals for compliance reports.

**Documentation Review**: Gather explanatory notes for documentation or training purposes.

**Quality Check**: Review cells with notes to identify items requiring attention.
<!-- END MANUAL -->

---

## Google Sheets Get Row

### What it is
Get a specific row by its index. Returns both list and dict formats.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a specific row from a Google Sheet by its 1-based row index using the Google Sheets API. The row data is returned both as a list (ordered values) and as a dictionary with header names as keys.

The dictionary format makes it easy to access specific fields by name rather than position.
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
**Record Retrieval**: Fetch a specific record by its row number for display or processing.

**Sequential Processing**: Get rows one at a time in a loop for individual processing.

**Verification**: Retrieve specific rows to verify data or confirm updates.
<!-- END MANUAL -->

---

## Google Sheets Get Row Count

### What it is
Get row count and dimensions of a Google Sheet. Useful for knowing where data ends.

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes a Google Sheet to determine its dimensions using the Google Sheets API. It returns total row count, data rows (excluding header), the last row with data, and column count.

This information is essential for determining loop boundaries or validating data presence.
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
**Loop Boundaries**: Determine how many rows to iterate through when processing data.

**Empty Check**: Verify if a sheet has data before attempting to process it.

**Progress Tracking**: Track how many records exist for reporting or dashboards.
<!-- END MANUAL -->

---

## Google Sheets Get Unique Values

### What it is
Get unique values from a column. Useful for building dropdown options or finding distinct categories.

### How it works
<!-- MANUAL: how_it_works -->
This block extracts unique values from a column in a Google Sheet using the Google Sheets API. Optionally includes counts for each value and can sort results by frequency.

Useful for discovering data categories, building dynamic dropdown lists, or analyzing data distribution.
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
**Dynamic Dropdowns**: Build dropdown options from actual data values in the sheet.

**Category Analysis**: Discover all unique categories or types in a dataset.

**Data Quality**: Identify unexpected values or typos by reviewing unique entries.
<!-- END MANUAL -->

---

## Google Sheets Import Csv

### What it is
Import CSV data into a Google Sheet

### How it works
<!-- MANUAL: how_it_works -->
This block imports CSV-formatted data into a Google Sheet using the Google Sheets API. You can specify the target sheet, starting cell, and whether to clear existing data first.

The CSV string is parsed and written to the sheet, enabling data import from external sources or API responses.
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
| result | Import result | Dict[str, Any] |
| rows_imported | Number of rows imported | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Import**: Import CSV data from external APIs or file uploads into Google Sheets.

**Bulk Updates**: Replace sheet data with fresh CSV exports from other systems.

**Migration**: Transfer data from CSV-based systems into Google Sheets.
<!-- END MANUAL -->

---

## Google Sheets Insert Row

### What it is
Insert a single row at a specific position. Existing rows shift down.

### How it works
<!-- MANUAL: how_it_works -->
This block inserts a row at a specific position in a Google Sheet using the Google Sheets API. Existing rows at and below the insertion point shift down to make room.

Use value_input_option to control whether values are parsed (USER_ENTERED) or stored as-is (RAW).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| row | Row values to insert (e.g., ['Alice', 'alice@example.com', '25']) | List[str] | Yes |
| row_index | 1-based row index where to insert (existing rows shift down) | int | Yes |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| value_input_option | How values are interpreted. USER_ENTERED: parsed like typed input (e.g., '=SUM(A1:A5)' becomes a formula, '1/2/2024' becomes a date). RAW: stored as-is without parsing. | "RAW" \| "USER_ENTERED" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the insert operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Priority Insertion**: Insert high-priority items at the top of lists.

**Ordered Data Entry**: Add new records at specific positions to maintain sorting.

**Template Rows**: Insert pre-formatted rows at designated positions.
<!-- END MANUAL -->

---

## Google Sheets List Named Ranges

### What it is
List all named ranges in a spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all named ranges defined in a Google Spreadsheet using the Google Sheets API. Returns each range's name, ID, and the cell range it references.

Useful for discovering available named ranges or auditing spreadsheet configuration.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| named_ranges | List of named ranges with name, id, and range info | List[Dict[str, Any]] |
| count | Number of named ranges | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Discovery**: List available named ranges to understand spreadsheet structure.

**Formula Building**: Get named range names for building formulas programmatically.

**Audit Documentation**: Generate documentation of named ranges for spreadsheet templates.
<!-- END MANUAL -->

---

## Google Sheets Lookup Row

### What it is
Look up a row by finding a value in a specific column. Returns the first matching row and optionally specific columns.

### How it works
<!-- MANUAL: how_it_works -->
This block performs a lookup operation similar to VLOOKUP—it finds a row where a specific column matches your search value, then returns that row's data. You can return all columns or just specific ones.

This is useful for database-style lookups where you need to find a record by ID, email, or any unique identifier.
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
**Customer Lookup**: Find customer details by email address or customer ID.

**Inventory Check**: Look up product information by SKU or product name.

**Configuration Retrieval**: Fetch settings for a specific entity from a configuration sheet.
<!-- END MANUAL -->

---

## Google Sheets Manage Sheet

### What it is
Create, delete, or copy sheets (sheet optional)

### How it works
<!-- MANUAL: how_it_works -->
This block manages sheets (tabs) within a Google Spreadsheet using the Google Sheets API. Operations include creating new sheets, deleting existing sheets, or copying sheets within the same spreadsheet.

Use this to dynamically organize spreadsheet structure as part of workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| operation | Operation to perform | "create" \| "delete" \| "copy" | Yes |
| sheet_name | Target sheet name (defaults to first sheet for delete) | str | No |
| source_sheet_id | Source sheet ID for copy | int | No |
| destination_sheet_name | New sheet name for copy | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Operation result | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Dynamic Structure**: Create new sheets for each month, project, or category as needed.

**Template Duplication**: Copy template sheets for new data entry cycles.

**Cleanup Operations**: Delete outdated or temporary sheets after processing.
<!-- END MANUAL -->

---

## Google Sheets Metadata

### What it is
This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties.

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves spreadsheet metadata using the Google Sheets API, including title, sheet names, sheet IDs, and properties like row/column counts and frozen rows.

Useful for understanding spreadsheet structure before performing operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The metadata of the spreadsheet including sheets info | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Structure Discovery**: Get sheet names and IDs before performing sheet-specific operations.

**Validation**: Verify spreadsheet structure matches expectations before data operations.

**Dynamic Routing**: Determine which sheets exist to route data appropriately.
<!-- END MANUAL -->

---

## Google Sheets Protect Range

### What it is
Protect a cell range or entire sheet from editing.

### How it works
<!-- MANUAL: how_it_works -->
This block adds protection to cell ranges or entire sheets using the Google Sheets API. Protected ranges can either block editing completely or show warnings while still allowing edits.

Use this to prevent accidental changes to important formulas, headers, or reference data.
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
| result | Result of the operation | Dict[str, Any] |
| protection_id | ID of the protection | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Formula Protection**: Protect cells containing critical formulas from accidental modification.

**Header Lock**: Prevent changes to header rows that could break data processing.

**Reference Data**: Protect lookup tables or configuration values from unauthorized changes.
<!-- END MANUAL -->

---

## Google Sheets Read

### What it is
A block that reads data from a Google Sheets spreadsheet using A1 notation range selection.

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
Remove duplicate rows based on specified columns. Keeps either the first or last occurrence.

### How it works
<!-- MANUAL: how_it_works -->
This block identifies and removes duplicate rows from a Google Sheet using the Google Sheets API. You can specify which columns to check for duplicates and whether to keep the first or last occurrence.

Case sensitivity is configurable for text comparisons.
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
| result | Result of the operation | Dict[str, Any] |
| removed_count | Number of duplicate rows removed | int |
| remaining_rows | Number of rows remaining | int |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Data Deduplication**: Clean up sheets containing duplicate entries from multiple data sources.

**Import Cleanup**: Remove duplicates after importing data from external systems.

**List Maintenance**: Keep email lists or contact lists free of duplicate entries.
<!-- END MANUAL -->

---

## Google Sheets Set Public Access

### What it is
Make a Google Spreadsheet public or private

### How it works
<!-- MANUAL: how_it_works -->
This block modifies the sharing settings of a Google Spreadsheet using the Google Drive API to make it publicly accessible or private. You can set the access level to reader or commenter.

When made public, anyone with the link can access the spreadsheet. The share link is returned for distribution.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to modify access for | Spreadsheet | No |
| public | True to make public, False to make private | bool | No |
| role | Permission role for public access | "reader" \| "commenter" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if operation failed | str |
| result | Result of the operation | Dict[str, Any] |
| share_link | Link to the spreadsheet | str |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Public Dashboards**: Make report spreadsheets publicly viewable for stakeholders.

**Resource Sharing**: Publish reference data or templates for public access.

**Access Control**: Toggle between public and private access based on workflow stages.
<!-- END MANUAL -->

---

## Google Sheets Share Spreadsheet

### What it is
Share a Google Spreadsheet with users or get shareable link

### How it works
<!-- MANUAL: how_it_works -->
This block shares a Google Spreadsheet with specific users by email address using the Google Drive API. You can set permission levels (reader, writer, commenter) and optionally send notification emails with custom messages.

Leave the email blank to just generate a shareable link.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | The spreadsheet to share | Spreadsheet | No |
| email | Email address to share with. Leave empty for link sharing. | str | No |
| role | Permission role for the user | "reader" \| "writer" \| "commenter" | No |
| send_notification | Send notification email to the user | bool | No |
| message | Optional message to include in notification email | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if share failed | str |
| result | Result of the share operation | Dict[str, Any] |
| share_link | Link to the spreadsheet | str |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Collaboration**: Share generated reports with stakeholders automatically.

**Team Onboarding**: Grant access to project spreadsheets for new team members.

**Client Delivery**: Share completed deliverables with clients including notification messages.
<!-- END MANUAL -->

---

## Google Sheets Sort

### What it is
Sort a Google Sheet by one or two columns. The sheet is sorted in-place.

### How it works
<!-- MANUAL: how_it_works -->
This block sorts data in a Google Sheet by one or two columns using the Google Sheets API. You can specify ascending or descending order for each column and whether to preserve a header row.

Sorting is performed in-place, modifying the sheet directly.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| sheet_name | Sheet name (optional, defaults to first sheet) | str | No |
| sort_column | Primary column to sort by (header name or column letter) | str | Yes |
| sort_order | Sort order for primary column | "ascending" \| "descending" | No |
| secondary_column | Secondary column to sort by (optional) | str | No |
| secondary_order | Sort order for secondary column | "ascending" \| "descending" | No |
| has_header | Whether the data has a header row (header won't be sorted) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the sort operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Report Organization**: Sort data by date, priority, or status before generating reports.

**Data Presentation**: Organize data alphabetically or numerically for better readability.

**Processing Order**: Sort items by priority to process high-priority items first.
<!-- END MANUAL -->

---

## Google Sheets Update Cell

### What it is
Update a single cell in a Google Sheets spreadsheet.

### How it works
<!-- MANUAL: how_it_works -->
This block updates a single cell in a Google Sheet using the Google Sheets API. Use A1 notation to specify the cell (e.g., "A1" or "Sheet1!B2").

Use value_input_option to control whether values are parsed (USER_ENTERED) or stored as-is (RAW).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet | Spreadsheet | No |
| cell | Cell address in A1 notation (e.g., 'A1', 'Sheet1!B2') | str | Yes |
| value | Value to write to the cell | str | Yes |
| value_input_option | How input data should be interpreted | "RAW" \| "USER_ENTERED" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | The result of the update operation | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Status Updates**: Update a status cell when processing completes.

**Timestamp Recording**: Write timestamps to cells marking when records were processed.

**Single Value Updates**: Change individual values like totals, flags, or configuration settings.
<!-- END MANUAL -->

---

## Google Sheets Update Row

### What it is
Update a specific row by its index. Can use list or dict format for values.

### How it works
<!-- MANUAL: how_it_works -->
This block updates an entire row in a Google Sheet by its 1-based row index using the Google Sheets API. You can provide values as an ordered list or as a dictionary with header names as keys.

The dictionary format is convenient when you only need to update specific columns.
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
| result | Result of the update operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Record Updates**: Update specific records identified by row index after lookup.

**Batch Field Updates**: Change multiple fields in a row simultaneously.

**Processing Results**: Write results back to source rows after processing.
<!-- END MANUAL -->

---

## Google Sheets Write

### What it is
A block that writes data to a Google Sheets spreadsheet at a specified A1 notation range.

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
| result | The result of the write operation | Dict[str, Any] |
| spreadsheet | The spreadsheet as a GoogleDriveFile (for chaining to other blocks) | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
An automated inventory system could use this block to update stock levels in a Google Sheets spreadsheet whenever products are sold or restocked, ensuring real-time inventory tracking.
<!-- END MANUAL -->

---
