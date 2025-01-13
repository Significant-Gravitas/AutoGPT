

# Basic Blocks Documentation

## Store Value Block

### What it is
A simple storage block that holds and forwards a value.

### What it does
Stores a provided value and makes it available for reuse without modification.

### How it works
Takes an input value and simply passes it through, allowing the same value to be used multiple times in different parts of your workflow.

### Inputs
- Input: The value to be stored (used when no data is provided)
- Data: A constant value to be retained (takes precedence over input)

### Outputs
- Output: The stored value

### Possible use case
When you need to reuse the same value multiple times in different parts of your workflow, such as a configuration setting or a common text string.

## Print To Console Block

### What it is
A debugging tool that displays text in the console.

### What it does
Prints provided text to the console for debugging purposes.

### How it works
Takes the input text and displays it in the console with a ">>>>> Print: " prefix.

### Inputs
- Text: The message to be printed to the console

### Outputs
- Status: Confirmation that the text was printed ("printed")

### Possible use case
When debugging your workflow and need to see the value of variables or confirm that certain steps are being executed.

## Find In Dictionary Block

### What it is
A lookup tool that searches for values in dictionaries, lists, or objects.

### What it does
Searches for a specified key in various data structures and returns the corresponding value.

### How it works
Takes an input structure (dictionary, list, or object) and a key, then attempts to find and return the associated value.

### Inputs
- Input: The data structure to search in
- Key: The key to look up

### Outputs
- Output: The found value
- Missing: The input data when the key is not found

### Possible use case
When you need to extract specific information from complex data structures, such as getting a user's email from a profile object.

## Agent Input Block

### What it is
A block that manages user input to your workflow.

### What it does
Handles input values with various options for configuration and validation.

### How it works
Processes input values with optional formatting, validation, and placeholder options.

### Inputs
- Name: Input identifier
- Value: The actual input value
- Title: Display title for the input
- Description: Detailed explanation of the input
- Placeholder Values: Suggested values
- Limit to Placeholder Values: Restricts input to suggested values only
- Advanced: Shows input in advanced section
- Secret: Handles sensitive information

### Outputs
- Result: The processed input value

### Possible use case
Creating a user form where you need to collect different types of information with specific validation rules.

## Agent Output Block

### What it is
A block that manages and formats workflow outputs.

### What it does
Records and optionally formats output values for users to view.

### How it works
Takes input values, applies optional formatting, and presents them as workflow outputs.

### Inputs
- Value: The data to output
- Name: Output identifier
- Title: Display title
- Description: Detailed explanation
- Format: Optional formatting template
- Advanced: Controls visibility in advanced section
- Secret: Handles sensitive information

### Outputs
- Output: The formatted or raw output value

### Possible use case
Displaying results of a data processing workflow with custom formatting.

## Add To Dictionary Block

### What it is
A utility for adding new entries to dictionaries.

### What it does
Adds one or more key-value pairs to an existing or new dictionary.

### How it works
Takes an existing dictionary (or creates a new one) and adds specified entries to it.

### Inputs
- Dictionary: Existing dictionary to modify
- Key: New entry's key
- Value: New entry's value
- Entries: Multiple key-value pairs to add

### Outputs
- Updated Dictionary: The modified dictionary
- Error: Any error messages

### Possible use case
Building a configuration object by adding settings one at a time.

## Add To List Block

### What it is
A utility for adding items to lists.

### What it does
Adds one or more items to an existing or new list at specified positions.

### How it works
Takes an existing list (or creates a new one) and adds entries at specified positions.

### Inputs
- List: Existing list to modify
- Entry: Single item to add
- Entries: Multiple items to add
- Position: Where to insert items

### Outputs
- Updated List: The modified list
- Error: Any error messages

### Possible use case
Building a task list by adding items at specific positions.

## Note Block

### What it is
A simple note display block.

### What it does
Displays text as a sticky note in the workflow.

### How it works
Takes input text and displays it as a visible note.

### Inputs
- Text: The content to display

### Outputs
- Output: The displayed text

### Possible use case
Adding documentation or reminders within your workflow.

## Create Dictionary Block

### What it is
A utility for creating new dictionaries.

### What it does
Creates a new dictionary with specified key-value pairs.

### How it works
Takes a set of key-value pairs and creates a new dictionary containing them.

### Inputs
- Values: Key-value pairs for the new dictionary

### Outputs
- Dictionary: The created dictionary
- Error: Any error messages

### Possible use case
Creating a new configuration object with predefined settings.

## Create List Block

### What it is
A utility for creating new lists.

### What it does
Creates a new list with specified values.

### How it works
Takes a set of values and creates a new list containing them.

### Inputs
- Values: Items to include in the new list

### Outputs
- List: The created list
- Error: Any error messages

### Possible use case
Creating a new collection of items, such as a list of tasks or settings.

