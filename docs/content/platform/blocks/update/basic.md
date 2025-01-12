
# Basic Blocks Documentation

## Store Value

### What it is
A simple block that stores and forwards values without modifying them.

### What it does
Takes an input value and makes it available for reuse multiple times without changes.

### How it works
Stores either a provided constant value or the input value and makes it available as output whenever needed.

### Inputs
- Input: The value to be stored when no constant data is provided
- Data: A constant value to be stored (optional)

### Outputs
- Output: The stored value (either from data or input)

### Possible use case
When you need to reuse a value multiple times in different parts of your workflow, such as a configuration setting or a common text string.

## Print To Console

### What it is
A debugging tool that displays text in the console.

### What it does
Prints the provided text to the console for debugging purposes.

### How it works
Takes the input text and displays it in the console with a ">>>>> Print:" prefix.

### Inputs
- Text: The message to be printed to the console

### Outputs
- Status: Confirmation that the text was printed ("printed")

### Possible use case
When debugging a workflow and need to see intermediate values or confirm that certain steps are being executed.

## Find In Dictionary

### What it is
A tool for retrieving values from dictionaries, lists, or objects.

### What it does
Looks up values using a specified key in various data structures.

### How it works
Searches for the provided key in the input data structure and returns the corresponding value if found.

### Inputs
- Input: The dictionary, list, or object to search in
- Key: The key to look up

### Outputs
- Output: The found value for the given key
- Missing: The original input if the key wasn't found

### Possible use case
When you need to extract specific information from complex data structures, such as getting a user's email from a user profile dictionary.

## Agent Input

### What it is
An interface block for receiving input values in a workflow.

### What it does
Provides a way to accept and validate input values with various configuration options.

### How it works
Creates an input interface with customizable settings like descriptions, placeholder values, and validation rules.

### Inputs
- Name: The identifier for the input
- Value: The actual input value
- Title: Display title for the input
- Description: Detailed explanation of the input
- Placeholder Values: Suggested values for the input
- Limit to Placeholder Values: Restricts input to suggested values only
- Advanced: Controls visibility in advanced settings
- Secret: Handles sensitive information

### Outputs
- Result: The processed input value

### Possible use case
Creating a user form where specific types of input are needed, such as a configuration interface.

## Agent Output

### What it is
A block for formatting and displaying workflow results.

### What it does
Processes and presents output values with optional formatting.

### How it works
Takes input values, applies optional formatting, and prepares them for display or further processing.

### Inputs
- Value: The data to be output
- Name: Identifier for the output
- Title: Display title
- Description: Detailed explanation
- Format: Template for formatting the output
- Advanced: Controls visibility in advanced settings
- Secret: Handles sensitive information

### Outputs
- Output: The formatted or raw output value

### Possible use case
Displaying results of a data processing workflow with custom formatting, such as creating a summary report.

[Documentation continues with remaining blocks...]

