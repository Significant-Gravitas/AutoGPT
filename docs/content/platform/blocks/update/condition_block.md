
## Condition Block

### What it is
A logical comparison tool that evaluates two values based on a specified operator and produces different outputs depending on whether the condition is true or false.

### What it does
This block compares two values using standard comparison operators (like equal to, greater than, less than) and outputs both the comparison result and corresponding values based on whether the condition was met or not.

### How it works
1. Takes two values and a comparison operator as input
2. Converts string numbers to actual numbers if possible
3. Performs the comparison using the selected operator
4. Outputs the result (true/false) and the corresponding value based on the result
5. If custom output values aren't provided, uses the input values as defaults

### Inputs
- Value 1: The first value to compare (can be a number, text, or boolean)
- Operator: The comparison operator to use (equals, not equals, greater than, less than, greater than or equal to, less than or equal to)
- Value 2: The second value to compare (can be a number, text, or boolean)
- Yes Value: (Optional) A custom value to output if the condition is true
- No Value: (Optional) A custom value to output if the condition is false

### Outputs
- Result: A boolean value (true/false) indicating whether the condition was met
- Yes Output: The value produced when the condition is true (either Yes Value or Value 1)
- No Output: The value produced when the condition is false (either No Value or Value 2)

### Possible use cases
1. Age verification: Compare a user's age with a minimum required age and output whether they can access content
2. Price comparison: Compare product prices and output whether something is within budget
3. Temperature control: Compare current temperature with a threshold and output different actions
4. Score evaluation: Compare test scores with passing grades and output pass/fail messages
5. Stock alerts: Compare current stock prices with target prices and output buy/sell signals

Example scenario:
A shopping app that compares a cart total ($45) with a free shipping threshold ($50). If the cart total is less than the threshold, it outputs "Add $5 more for free shipping"; if it's greater than or equal to the threshold, it outputs "Free shipping eligible!"

