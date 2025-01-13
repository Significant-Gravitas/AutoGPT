
<file_name>autogpt_platform/backend/backend/blocks/branching.md</file_name>

## Condition Block

### What it is
A logic block that performs comparisons between two values and returns results based on the comparison outcome.

### What it does
This block evaluates a condition by comparing two values using a specified operator (like equals, greater than, etc.) and provides different outputs based on whether the condition is true or false.

### How it works
The block takes two values and compares them using the selected operator. If the condition is true, it outputs the "yes_value" (or the first value if no "yes_value" is specified). If the condition is false, it outputs the "no_value" (or the second value if no "no_value" is specified).

### Inputs
- Value 1: The first value for comparison (can be a number, text, or boolean)
- Operator: The comparison operator to use (equals, not equals, greater than, less than, greater than or equal to, less than or equal to)
- Value 2: The second value for comparison (can be a number, text, or boolean)
- Yes Value (Optional): The value to output if the condition is true (defaults to Value 1 if not specified)
- No Value (Optional): The value to output if the condition is false (defaults to Value 2 if not specified)

### Outputs
- Result: A boolean value (true/false) indicating whether the condition was met
- Yes Output: The value returned when the condition is true
- No Output: The value returned when the condition is false

### Possible use cases
- Creating an age verification system that outputs "allowed" if age is 18 or above, and "not allowed" if below 18
- Building a price comparison tool that outputs "buy" if the price is below a threshold, and "wait" if it's above
- Implementing a temperature control system that activates cooling when temperature is above a certain point
- Creating a grading system that outputs "pass" or "fail" based on a score threshold
- Building a stock trading logic that makes decisions based on price comparisons

