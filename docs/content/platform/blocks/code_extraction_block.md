
## CSV Reader

### What it is
A data processing tool that reads and interprets CSV (Comma-Separated Values) files, converting them into a more usable format.

### What it does
This block reads CSV file contents and transforms them into structured data, processing the file row by row. It can handle different file formats and configurations, and outputs both individual rows and the complete dataset.

### How it works
The block takes CSV content as text and processes it according to specified settings. It first checks for a header row if specified, then reads through each line of the file, organizing the data into a structured format. It can skip certain rows or columns and clean up the data by removing extra spaces if needed.

### Inputs
- Contents: The actual content of the CSV file that needs to be processed
- Delimiter: The character that separates values in your CSV file (default is comma ",")
- Quote Character: The character used to enclose special values (default is double quote """)
- Escape Character: The character used to handle special characters in the file (default is backslash "\")
- Has Header: Whether the first row contains column names (default is Yes)
- Skip Rows: Number of rows to skip from the beginning of the file (default is 0)
- Strip: Whether to remove extra spaces from values (default is Yes)
- Skip Columns: List of column names to exclude from the output (default is none)

### Outputs
- Row: Individual rows of data from the CSV file, presented as key-value pairs where keys are either column headers or column numbers
- All Data: The complete dataset as a list of all processed rows

### Possible use case
A business analyst needs to process monthly sales reports that come in CSV format. The reports have a header row and some empty rows at the top that need to be skipped. The analyst can use this block to:
1. Load the CSV file
2. Skip the unnecessary rows
3. Automatically recognize column names from the header
4. Get clean, structured data for further analysis
5. Process either individual rows for immediate actions or work with the complete dataset for overall analysis

This block would handle all the data preparation automatically, saving time and reducing the chance of errors from manual data handling.

