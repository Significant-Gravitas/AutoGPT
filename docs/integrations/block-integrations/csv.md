# Read CSV

## What It Is <a href="#what-it-is" id="what-it-is"></a>

A block that reads and processes CSV (Comma-Separated Values) files.

## What It Does <a href="#what-it-does" id="what-it-does"></a>

This block takes CSV content as input, processes it, and outputs the data as individual rows and a complete dataset.

## How It Works <a href="#how-it-works" id="how-it-works"></a>

The Read CSV block takes the contents of a CSV file and splits it into rows and columns. It can handle different formatting options, such as custom delimiters and quote characters. The block processes the CSV data and outputs each row individually, as well as the complete dataset.

## Inputs <a href="#inputs" id="inputs"></a>

| Input         | Description                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------ |
| Contents      | The CSV data as a string                                                                         |
| Delimiter     | The character used to separate values in the CSV (default is comma ",")                          |
| Quotechar     | The character used to enclose fields containing special characters (default is double quote '"') |
| Escapechar    | The character used to escape special characters (default is backslash "")                        |
| Has\_header   | Indicates whether the CSV has a header row (default is true)                                     |
| Skip\_rows    | The number of rows to skip at the beginning of the CSV (default is 0)                            |
| Strip         | Whether to remove leading and trailing whitespace from values (default is true)                  |
| Skip\_columns | A list of column names to exclude from the output (default is an empty list)                     |

## Outputs <a href="#outputs" id="outputs"></a>

| Output    | Description                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------ |
| Row       | A dictionary representing a single row of the CSV, with column names as keys and cell values as values |
| All\_data | A list of dictionaries containing all rows from the CSV                                                |

## Possible Use Case <a href="#possible-use-case" id="possible-use-case"></a>

This block could be used in a data analysis pipeline to import and process customer information from a CSV file. The individual rows could be used for real-time processing, while the complete dataset could be used for batch analysis or reporting.
