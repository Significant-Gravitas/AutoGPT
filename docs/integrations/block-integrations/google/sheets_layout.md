# Google Sheets Layout
<!-- MANUAL: file_description -->
Blocks that lay out a Google Sheet so it is easy to read: frozen header rows and columns, column widths, wrapped text, filter buttons and alternating row colours.
<!-- END MANUAL -->

## Google Sheets Set Layout

### What it is
Lay out a Google Sheet: freeze header rows or columns, set column widths, wrap text, add filter buttons and alternating row colours.

### How it works
<!-- MANUAL: how_it_works -->
The block reads the spreadsheet's sheets once, then sends one Sheets API batchUpdate with only the options you set: frozen rows and columns, a pixel width per column from A onwards (up to 2000 px, at most 200 columns), wrapped and top-aligned text on a range, filter buttons on a range, and alternating row colours on a range. It works on `sheet_name`, or on the sheet the ranges name, or else the first sheet, and refuses a range that names a different sheet. Before adding alternating colours, the block removes every existing set of alternating colours that overlaps the new range. Each is removed in full, so if one reaches beyond the new range, that part loses its colours too. Sets that don't overlap stay, and running the block again does not fail.

A spreadsheet must be picked: the field is optional in the schema only because the Drive picker fills it, and without one the block returns "No spreadsheet selected".
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spreadsheet | Select a Google Sheets spreadsheet. At runtime, feed this from an AgentGoogleDriveFileInputBlock with matching allowed_views. NEVER hardcode a file ID in input_default (including one parsed from a Drive URL the user pasted in chat) — only the picker attaches the _credentials_id needed for auth. | Spreadsheet | No |
| sheet_name | Sheet to lay out. Optional: defaults to the sheet the ranges name, else the first sheet | str | No |
| frozen_rows | Rows to keep in view at the top, e.g. 1 for a header row. -1 leaves it as is. | int | No |
| frozen_columns | Columns to keep in view on the left. -1 leaves it as is. | int | No |
| column_widths | Widths in pixels for columns A, B, C and on, up to 2000. 0 leaves that column as is. | List[int] | No |
| wrap_text_range | Range whose text wraps inside the cell, aligned to the top, e.g. 'A1:L1000' | str | No |
| filter_range | Range to add filter buttons to, header row first, e.g. 'A1:L1000' | str | No |
| banding_range | Range to give alternating row colours, header row first, e.g. 'A1:L1000'. Any existing alternating colours that overlap it are removed in full, including their part outside this range. | str | No |
| header_color | Header row colour for the alternating colours | SheetColor | No |
| band_color | Colour of every second row | SheetColor | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if any | str |
| result | Result of the operation | Dict[str, Any] |
| spreadsheet | The spreadsheet for chaining | GoogleDriveFile |

### Possible use case
<!-- MANUAL: use_case -->
**Lead tracker**: After creating a sheet and writing its header row, freeze the header, size the columns for names, notes and links, wrap the notes, and add filters and alternating colours so the sheet is easy to read and sort.
<!-- END MANUAL -->

---
