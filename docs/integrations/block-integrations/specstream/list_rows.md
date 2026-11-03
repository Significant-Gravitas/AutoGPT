# Specstream List Rows
<!-- MANUAL: file_description -->
Blocks that read the rows of a SpecStream run.
<!-- END MANUAL -->

## SpecStream List Rows

### What it is
Reads the rows of a SpecStream dataset run. Source URL and read time on every value.

### How it works
<!-- MANUAL: how_it_works -->
The block reads the rows of a run page by page and returns each row with its values by field. Each value comes with its source URL and read time, including the details read in words and the Matches cell. A value that SpecStream built rather than read from a page, such as a likely email address, has an empty source URL. While a run is still going, the rows found so far are returned. Use next_cursor to read the following rows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | The run ID from Approve and Run | str | Yes |
| max_rows | The most rows to return in this call | int | No |
| cursor | The next_cursor of an earlier call, to read the following rows | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| row | One row: its values by field, and each value with its source URL and read time | DatasetRow |
| rows | All the rows of this call | List[DatasetRow] |
| row_count | How many rows this call returned | int |
| status | The status of the run. While it runs, the rows found so far are returned. | str |
| next_cursor | Pass this as cursor to read the following rows. Empty when there are no more. | str |

### Possible use case
<!-- MANUAL: use_case -->
Send the rows to a spreadsheet or a CRM step, and keep the source URL and read time next to each value.
<!-- END MANUAL -->

---
