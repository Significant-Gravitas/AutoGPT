

# Mathematical Operations Blocks Documentation

## Calculator

### What it is
A versatile mathematical calculator that performs basic arithmetic operations between two numbers.

### What it does
Performs one of five mathematical operations (addition, subtraction, multiplication, division, or power) on two numbers and provides the result. It also includes an option to round the result to the nearest whole number.

### How it works
The calculator takes two numbers and a selected operation as input. It then performs the chosen mathematical operation on these numbers and returns the result. If division by zero is attempted, it returns infinity. For any other errors, it returns "NaN" (Not a Number).

### Inputs
- Operation: Choose from Add, Subtract, Multiply, Divide, or Power
- Number A: The first number in the calculation
- Number B: The second number in the calculation
- Round Result: Toggle whether you want the result rounded to the nearest whole number

### Outputs
- Result: The final calculated value from the operation

### Possible use case
A financial application that needs to perform various calculations, such as computing interest rates, calculating discounts, or determining payment installments.

## Count Items

### What it is
A counter that determines the number of items in any collection of data.

### What it does
Counts and returns the total number of items in various types of collections, such as lists, dictionaries, strings, or any other countable group of items.

### How it works
The block examines the provided collection and counts its elements. It can handle different types of collections (strings, lists, dictionaries, etc.) and returns the total count. If it encounters an error or receives an invalid input, it returns -1.

### Inputs
- Collection: The group of items you want to count (can be a list, dictionary, string, or any other collection of items)

### Outputs
- Count: The total number of items found in the collection

### Possible use case
An inventory management system that needs to quickly count items in different categories, or a text analysis tool that needs to count characters, words, or sentences in a document.

