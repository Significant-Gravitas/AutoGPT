
## Step Through Items

### What it is
A logic block that helps you process lists or dictionaries one item at a time in a systematic way.

### What it does
This block takes a collection of items (either a list or dictionary) and processes them one at a time, outputting each item along with its position (index or key) in sequence.

### How it works
The block accepts input data in three different formats (list, dictionary, or string representation) and processes them systematically:
1. For lists: It goes through each item in order, providing both the item and its position number
2. For dictionaries: It processes each value in the dictionary, outputting both the value and its corresponding key
3. For string inputs: It first converts the string into either a list or dictionary (if properly formatted) and then processes it accordingly

### Inputs
- items: A list of items to process (e.g., [1, 2, 3, 4, 5])
- items_object: A dictionary of items to process (e.g., {"key1": "value1", "key2": "value2"})
- items_str: A string representation of either a list or dictionary to process

### Outputs
- item: The current item being processed from the collection
- key: The position (index number for lists) or identifier (key for dictionaries) of the current item

### Possible use cases
1. Processing a list of customer orders one at a time, where each order needs individual attention
2. Analyzing survey responses sequentially, where each response needs to be evaluated independently
3. Processing a collection of data entries where each entry requires specific validation or transformation
4. Handling a series of tasks that need to be executed in sequence
5. Managing a queue of messages or notifications that need to be processed one at a time

### Notes
- The block is particularly useful when you need to perform actions on each item in a collection individually
- It can handle both simple lists (like numbers or text) and complex data structures (like nested dictionaries)
- The block will automatically skip any empty inputs, making it flexible for various use cases
- It's part of the LOGIC category of blocks, indicating its utility in controlling program flow and data processing
