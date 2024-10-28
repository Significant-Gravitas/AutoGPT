## Step Through Items

### What it is
A block that iterates through a list or dictionary, processing each item one by one.

### What it does
This block takes a list or dictionary as input and goes through each item, outputting the current item and its corresponding key or index.

### How it works
When given a list or dictionary, the block processes each item individually. For lists, it keeps track of the item's position (index). For dictionaries, it focuses on the values, using the value as both the item and the key in the output.

### Inputs
| Input | Description |
|-------|-------------|
| Items | A list or dictionary that you want to process item by item. For example, you could input a list of numbers [1, 2, 3, 4, 5] or a dictionary of key-value pairs {'key1': 'value1', 'key2': 'value2'} |

### Outputs
| Output | Description |
|--------|-------------|
| Item | The current item being processed from the input list or dictionary |
| Key | For lists, this is the index (position) of the current item. For dictionaries, this is the same as the item (the dictionary's value) |

### Possible use case
Imagine you have a list of customer names and you want to perform a specific action for each customer, like sending a personalized email. This block could help you go through the list one by one, allowing you to process each customer individually.