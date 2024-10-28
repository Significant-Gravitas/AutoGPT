## Google Sheets Read

### What it is
A block that reads data from a Google Sheets spreadsheet.

### What it does
This block retrieves information from a specified range within a Google Sheets spreadsheet.

### How it works
The block connects to Google Sheets using provided credentials, then fetches data from the specified spreadsheet and range.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Authentication information required to access Google Sheets |
| Spreadsheet ID | The unique identifier of the spreadsheet you want to read from |
| Range | The specific area of the spreadsheet you want to read (e.g., "Sheet1!A1:B2") |

### Outputs
| Output | Description |
|--------|-------------|
| Result | The data retrieved from the spreadsheet, organized in rows and columns |
| Error | Any error message that occurred during the process |

### Possible use case
A marketing team could use this block to automatically retrieve the latest campaign performance data from a shared Google Sheets document for analysis and reporting.

---

## Google Sheets Write

### What it is
A block that writes data to a Google Sheets spreadsheet.

### What it does
This block allows you to input data into a specified range within a Google Sheets spreadsheet.

### How it works
The block authenticates with Google Sheets using provided credentials, then updates the specified spreadsheet range with the given data.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Authentication information required to access Google Sheets |
| Spreadsheet ID | The unique identifier of the spreadsheet you want to write to |
| Range | The specific area of the spreadsheet where you want to write data (e.g., "Sheet1!A1:B2") |
| Values | The data you want to write to the spreadsheet, organized in rows and columns |

### Outputs
| Output | Description |
|--------|-------------|
| Result | Information about the write operation, such as the number of cells, columns, and rows updated |
| Error | Any error message that occurred during the process |

### Possible use case
An automated inventory system could use this block to update stock levels in a Google Sheets spreadsheet whenever products are sold or restocked, ensuring real-time inventory tracking.