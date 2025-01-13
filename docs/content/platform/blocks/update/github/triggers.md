
<file_name>autogpt_platform/backend/backend/blocks/google/sheets.md</file_name>

# Google Sheets Integration Blocks

## Google Sheets Read

### What it is
A block that enables reading data from Google Sheets spreadsheets.

### What it does
Retrieves data from a specified range within a Google Sheets spreadsheet and returns it as a structured list.

### How it works
The block connects to Google Sheets using provided credentials, locates the specified spreadsheet using its ID, and extracts data from the designated range using A1 notation (e.g., "Sheet1!A1:B2").

### Inputs
- Credentials: Google OAuth credentials required to access the spreadsheet
- Spreadsheet ID: The unique identifier of the Google Sheets document you want to read from
- Range: The specific cell range you want to read (using A1 notation)

### Outputs
- Result: A list of lists containing the data read from the spreadsheet, where each inner list represents a row of data
- Error: Any error message that occurred during the operation

### Possible use case
A business analyst could use this block to automatically retrieve weekly sales data stored in a Google Sheets document for further processing or analysis.

## Google Sheets Write

### What it is
A block that enables writing data to Google Sheets spreadsheets.

### What it does
Writes data to a specified range within a Google Sheets spreadsheet and returns information about the update operation.

### How it works
The block authenticates with Google Sheets using provided credentials, locates the target spreadsheet, and writes the provided data to the specified range using A1 notation.

### Inputs
- Credentials: Google OAuth credentials required to access and modify the spreadsheet
- Spreadsheet ID: The unique identifier of the Google Sheets document you want to write to
- Range: The specific cell range where you want to write the data (using A1 notation)
- Values: The data to write, formatted as a list of lists where each inner list represents a row

### Outputs
- Result: A dictionary containing information about the write operation, including:
  - Number of cells updated
  - Number of columns updated
  - Number of rows updated
- Error: Any error message that occurred during the operation

### Possible use case
An automated reporting system could use this block to write processed data or analysis results directly to a shared Google Sheets document that team members can access and review.
