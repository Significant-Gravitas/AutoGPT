
## Read CSV

### What it is
A data processing tool that reads and interprets CSV (Comma-Separated Values) files, converting them into a more usable format.

### What it does
This block takes CSV file contents and transforms them into structured data, making it easier to work with spreadsheet-like information. It can process the data row by row and also provide all the data at once.

### How it works
The block reads the provided CSV content, processes it according to the specified settings (like delimiters and headers), and converts each row into a dictionary format where column names (or numbers) are paired with their corresponding values. It can either output the data one row at a time or provide all rows at once.

### Inputs
- Contents: The actual content of the CSV file you want to process
- Delimiter: The character that separates values in your CSV file (default is comma ",")
- Quote Character: The character used to enclose special values (default is double quote """)
- Escape Character: The character used to handle special characters in the file (default is backslash "\")
- Has Header: Whether your CSV file includes a header row with column names (default is Yes)
- Skip Rows: Number of rows to skip from the beginning of the file (default is 0)
- Strip: Whether to remove extra spaces from the values (default is Yes)
- Skip Columns: List of column names to exclude from the output (default is none)

### Outputs
- Row: Individual rows of data, where each row is converted into a dictionary with column names (or numbers) as keys and cell values as values
- All Data: The complete dataset as a list of dictionaries, where each dictionary represents one row of the CSV file

### Possible use case
A data analyst needs to process a large customer database exported as a CSV file. They want to:
1. Remove unnecessary whitespace from all entries
2. Skip the first two rows containing metadata
3. Exclude sensitive columns like "social_security_number"
4. Process the data both row by row for immediate analysis and as a complete dataset for backup

Using this Read CSV block, they can easily configure these requirements and receive the data in a clean, structured format ready for further analysis or processing.

