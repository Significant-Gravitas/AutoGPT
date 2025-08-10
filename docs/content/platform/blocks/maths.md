# Mathematical Operations Blocks

## Calculator

### What it is
A block that performs mathematical operations on two numbers.

### What it does
This block takes two numbers and performs a selected mathematical operation (addition, subtraction, multiplication, division, or exponentiation) on them. It can also optionally round the result to a whole number.

### How it works
The Calculator block takes in two numbers and an operation choice. It then applies the chosen operation to the numbers and returns the result. If rounding is selected, it rounds the result to the nearest whole number.

### Inputs
| Input | Description |
|-------|-------------|
| Operation | Choose the math operation you want to perform (Add, Subtract, Multiply, Divide, or Power) |
| A | Enter the first number for the calculation |
| B | Enter the second number for the calculation |
| Round result | Choose whether to round the result to a whole number (True or False) |

### Outputs
| Output | Description |
|--------|-------------|
| Result | The result of your calculation |

### Possible use case
A user wants to quickly perform a calculation, such as adding two numbers or calculating a percentage. They can input the numbers and operation into this block and receive the result instantly.

---

## Count Items

### What it is
A block that counts the number of items in a collection.

### What it does
This block takes a collection (such as a list, dictionary, or string) and counts the number of items within it.

### How it works
The Count Items block receives a collection as input. It then determines the type of collection and uses the appropriate method to count the items. For most collections, it uses the length function. For other iterable objects, it counts the items one by one.

### Inputs
| Input | Description |
|-------|-------------|
| Collection | Enter the collection you want to count (e.g., a list, dictionary, or string) |

### Outputs
| Output | Description |
|--------|-------------|
| Count | The number of items in the collection |

### Possible use case
A user has a list of customer names and wants to quickly determine how many customers are in the list. They can input the list into this block and receive the total count immediately.