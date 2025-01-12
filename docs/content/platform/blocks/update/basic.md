

# Basic Blocks Documentation

## Store Value

### What it is
A simple block that stores and forwards a value without modifying it.

### What it does
Takes an input value and makes it available for reuse multiple times without changing it.

### How it works
Stores either a provided constant value or the input value and outputs it whenever requested.

### Inputs
- Input: The value to be stored if no data is provided
- Data: A constant value to be stored (optional)

### Outputs
- Output: The stored value (either from data or input)

### Possible use case
When you need to reuse the same value multiple times in different parts of your workflow, like a configuration setting or a constant value.

## Print To Console

### What it is
A debugging tool that displays text in the console.

### What it does
Prints the provided text to the console for debugging purposes.

### How it works
Takes the input text and displays it in the console with a prefix ">>>>> Print: "

### Inputs
- Text: The message to be printed to the console

### Outputs
- Status: Confirmation that the text was printed ("printed")

### Possible use case
When debugging your workflow and need to see the value of variables or confirm that certain steps are being executed.

## Find In Dictionary

### What it is
A tool for looking up values in dictionaries, lists, or objects.

### What it does
Searches for a specified key in various data structures and returns the corresponding value.

### How it works
Examines the input data structure and attempts to find the requested key, returning either the found value or indicating that it's missing.

### Inputs
- Input: The data structure to search in (dictionary, list, or object)
- Key: The key or index to look up

### Outputs
- Output: The found value for the given key
- Missing: The input data if the key wasn't found

### Possible use case
When working with complex data structures and need to extract specific pieces of information.

## Agent Input

### What it is
A block for handling user input in workflows.

### What it does
Manages and processes input values with various options for customization.

### How it works
Takes a value along with metadata about how it should be handled and makes it available to the workflow.

### Inputs
- Name: Identifier for the input
- Value: The actual input value
- Title: Display title (optional)
- Description: Explanatory text (optional)
- Placeholder Values: Suggested values (optional)
- Limit to Placeholder Values: Restricts input to suggested values
- Advanced: Whether to show in advanced settings
- Secret: Whether to treat as sensitive information

### Outputs
- Result: The processed input value

### Possible use case
Creating a user interface where users need to provide information with specific constraints or guidance.

## Agent Output

### What it is
A block for formatting and displaying workflow results.

### What it does
Takes values and formats them for presentation as output.

### How it works
Processes the input value using optional formatting rules and prepares it for display.

### Inputs
- Value: The data to be output
- Name: Identifier for the output
- Title: Display title (optional)
- Description: Explanatory text (optional)
- Format: Template for formatting the output
- Advanced: Whether to treat as advanced output
- Secret: Whether to treat as sensitive information

### Outputs
- Output: The formatted result

### Possible use case
Creating formatted reports or displaying results in a specific way.

## Add To Dictionary

### What it is
A tool for adding new entries to dictionaries.

### What it does
Adds one or more key-value pairs to an existing or new dictionary.

### How it works
Takes a dictionary (or creates a new one) and adds specified entries to it.

### Inputs
- Dictionary: Existing dictionary to modify (optional)
- Key: New key to add
- Value: Value for the new key
- Entries: Multiple key-value pairs to add

### Outputs
- Updated Dictionary: The dictionary with new entries
- Error: Any error messages

### Possible use case
Building or modifying configuration settings or collecting data incrementally.

[Documentation continues with remaining blocks...]

