
## Step Through Items

### What it is
A logical processing block that helps you work through lists or collections of items one at a time, like flipping through pages in a book.

### What it does
This block takes a collection of items (which could be a list, dictionary, or text representation of either) and processes them one by one, outputting each item along with its position (key or index) in the sequence.

### How it works
The block accepts your collection of items and then:
1. Checks which input format you've provided (list, dictionary, or text)
2. If you've provided text, it converts it into a proper list or dictionary
3. Goes through each item one by one
4. For each item, it produces two pieces of information: the item itself and its position
5. If working with a dictionary, it focuses on the values rather than the keys

### Inputs
- Items (List): A collection of items in list format, like a shopping list or sequence of numbers
- Items Object (Dictionary): A collection of paired items, like a contact list with names and phone numbers
- Items String (Text): A text version of either a list or dictionary, useful when your data comes from text sources

### Outputs
- Item: The current piece of data being processed from your collection
- Key: The position or identifier of the current item (will be a number for lists, starting from 0)

### Possible use case
Imagine you have a list of customer orders that need to be processed one at a time. This block could help you:
1. Take the list of orders
2. Process each order individually
3. Keep track of which order number you're working on
4. Ensure no orders are missed or processed multiple times

For example, if you had a list of customer names `["Alice", "Bob", "Charlie"]`, the block would output:
- First round: Item="Alice", Key=0
- Second round: Item="Bob", Key=1
- Third round: Item="Charlie", Key=2

This makes it perfect for situations where you need to perform the same action on multiple items in sequence while keeping track of your progress.
