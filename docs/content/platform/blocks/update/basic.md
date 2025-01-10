
## Condition Block

### What it is
A logical comparison tool that evaluates two values based on a specified operator and produces different outputs depending on whether the condition is true or false.

### What it does
This block compares two values using standard comparison operators (like equals, greater than, less than) and outputs both the result of the comparison and corresponding values based on whether the condition was met or not.

### How it works
The block takes two values and compares them using the selected operator. It then determines if the comparison is true or false. Based on this result, it outputs either a "yes value" (if true) or a "no value" (if false). If no specific yes/no values are provided, it uses the input values as defaults.

### Inputs
- Value 1: The first value to compare (can be a number, text, or boolean)
- Operator: The comparison operator to use (equals, not equals, greater than, less than, greater than or equal to, less than or equal to)
- Value 2: The second value to compare (can be a number, text, or boolean)
- Yes Value (Optional): The value to output if the comparison is true (if not specified, Value 1 will be used)
- No Value (Optional): The value to output if the comparison is false (if not specified, Value 2 will be used)

### Outputs
- Result: A boolean value (true/false) indicating whether the condition was met
- Yes Output: The value produced when the condition is true (either the specified Yes Value or Value 1)
- No Output: The value produced when the condition is false (either the specified No Value or Value 2)

### Possible use case
In an e-commerce application, this block could be used to implement a discount system:
- Value 1: Customer's purchase amount
- Operator: Greater than
- Value 2: $100
- Yes Value: "10% discount applied"
- No Value: "No discount available"

This would output different messages based on whether the purchase amount exceeds $100, helping to implement a conditional discount policy.

