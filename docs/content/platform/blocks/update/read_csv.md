
## CSV Reader

### What it is
A tool that reads and processes CSV (Comma-Separated Values) files, converting them into a more usable format.

### What it does
This block takes CSV-formatted text and transforms it into structured data, processing it row by row while allowing customization of how the CSV is read and interpreted. It can handle different file formats and configurations, making it versatile for various data processing needs.

### How it works
The block reads a CSV file's contents line by line, organizing the data into a structured format. It can recognize headers, skip specific rows or columns, and handle different types of delimiters and special characters. As it processes the file, it produces both individual rows and a complete collection of all the data.

### Inputs
- Contents: The actual CSV text data you want to process
- Delimiter: The character that separates values in your CSV (defaults to comma)
- Quote Character: The character used to enclose text fields (defaults to double quote)
- Escape Character: The character used to escape special characters (defaults to backslash)
- Has Header: Whether your CSV includes a header row (defaults to yes)
- Skip Rows: Number of rows to skip at the beginning of the file (defaults to 0)
- Strip: Whether to remove extra spaces from values (defaults to yes)
- Skip Columns: List of columns you want to exclude from the results (defaults to none)

### Outputs
- Row: Each individual row of data as it's processed, presented as a collection of key-value pairs
- All Data: The complete set of all rows from the CSV file, presented as a list of key-value pair collections

### Possible use case
A data analyst needs to process customer information stored in a CSV file. The file contains customer names, addresses, and purchase history, but has some extra header rows and unnecessary columns. Using this block, they can:
1. Skip the extra header rows
2. Exclude sensitive columns like customer IDs
3. Clean up the data by removing extra spaces
4. Process the information either row by row for immediate analysis or as a complete dataset for bulk processing
5. Use the structured output for further analysis or integration with other systems

The block would handle all the complexities of reading and parsing the CSV file while providing clean, structured data ready for use in their analysis.

