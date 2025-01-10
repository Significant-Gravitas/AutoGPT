
## Read CSV

### What it is
A data processing block that reads and parses CSV (Comma-Separated Values) files, converting them into a more usable format.

### What it does
This block takes CSV-formatted text and transforms it into structured data, processing it row by row. It can handle various CSV formats and configurations, making it versatile for different types of CSV files.

### How it works
The block reads a CSV file's contents, splits it into rows and columns based on specified delimiters, and converts each row into a dictionary where column headers (or numbers if no headers exist) become keys and cell contents become values. It can process the file in two ways: outputting data row by row and providing all data at once.

### Inputs
- Contents: The actual text content of the CSV file to be processed
- Delimiter: The character that separates values in your CSV file (default is comma ",")
- Quote Character: The character used to enclose fields that contain special characters (default is double quote """)
- Escape Character: The character used to escape special characters in the CSV (default is backslash "\")
- Has Header: Indicates whether the first row contains column names (default is Yes)
- Skip Rows: Number of rows to skip from the beginning of the file (default is 0)
- Strip: Whether to remove extra whitespace from values (default is Yes)
- Skip Columns: List of column names to exclude from the output (default is none)

### Outputs
- Row: Provides each row of the CSV as a dictionary, where keys are either column headers or column indices
- All Data: Delivers the complete CSV data as a list of dictionaries, where each dictionary represents one row

### Possible use case
A data analyst needs to process a monthly sales report that comes in CSV format. The CSV file contains columns for date, product, quantity, and price. Using this block, they can:
1. Load the CSV file
2. Automatically recognize column headers
3. Process the data row by row for immediate analysis
4. Get a complete dataset for overall reporting
5. Skip unnecessary columns like internal reference numbers
6. Clean up any extra whitespace in the data

The block would be particularly useful when dealing with large CSV files, as it can process data row by row without loading everything into memory at once, and it provides flexibility in handling different CSV formats and structures.

