
<file_name>autogpt_platform/backend/backend/blocks/iteration.md</file_name>

## Step Through Items

### What it is
A logical operation block that helps you process lists or dictionaries one item at a time.

### What it does
This block takes a collection of items (either a list or dictionary) and iterates through each element, outputting one item at a time along with its corresponding position (index or key).

### How it works
The block accepts input in three different formats (list, dictionary, or string representation of either) and processes them sequentially. For each item in the collection, it outputs two pieces of information: the current item and its position in the collection. If the input is a dictionary, it processes the values; if it's a list, it processes each element while keeping track of their positions.

### Inputs
- Items (list): A list of elements to iterate through (e.g., [1, 2, 3, 4, 5])
- Items Object (dictionary): A dictionary of key-value pairs to iterate through (e.g., {'key1': 'value1', 'key2': 'value2'})
- Items String (string): A text representation of either a list or dictionary to iterate through

### Outputs
- Item: The current element being processed from the collection
- Key: The position (index number for lists) or identifier (key for dictionaries) of the current item

### Possible use case
Imagine you have a list of customer names and want to process each one individually, perhaps to send personalized emails. This block would help by providing one customer name at a time, along with their position in the list, allowing you to track progress and process each customer separately.

For example, with a list like ["John", "Mary", "Steve"], the block would output:
1. Item: "John", Key: 0
2. Item: "Mary", Key: 1
3. Item: "Steve", Key: 2

This sequential processing makes it easier to manage large collections of data by breaking them down into individual pieces that can be handled one at a time.

