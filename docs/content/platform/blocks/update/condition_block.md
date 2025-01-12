
## Condition Block

### What it is
A logical comparison tool that evaluates two values based on a specified operator and produces different outputs depending on whether the condition is true or false.

### What it does
This block compares two values using standard comparison operators (equal to, greater than, less than, etc.) and outputs both the result of the comparison and corresponding values based on whether the condition was met.

### How it works
1. Takes two values and a comparison operator as input
2. Converts string numbers to actual numbers when possible
3. Performs the specified comparison between the values
4. Outputs whether the condition was true or false
5. Provides different output values based on the comparison result

### Inputs
- Value 1: The first value to compare (can be a number, text, or boolean)
- Operator: The comparison operator to use (equals, not equals, greater than, less than, greater than or equal to, less than or equal to)
- Value 2: The second value to compare (can be a number, text, or boolean)
- Yes Value: (Optional) A specific value to output if the condition is true. If not provided, Value 1 will be used
- No Value: (Optional) A specific value to output if the condition is false. If not provided, Value 2 will be used

### Outputs
- Result: A boolean (true/false) indicating whether the condition was met
- Yes Output: The value returned when the condition is true (either Yes Value or Value 1)
- No Output: The value returned when the condition is false (either No Value or Value 2)

### Possible use cases
1. Age verification system: Compare a user's age against a minimum requirement and output different messages
2. Price comparison: Check if a product's price is below a certain threshold and display different labels
3. Temperature control: Compare current temperature with desired temperature and output different actions
4. Score evaluation: Compare test scores against passing grades and output different feedback messages
5. Inventory management: Compare stock levels against minimum threshold and output different alert messages

For example, you could use this block to:
- Compare if a temperature (Value 1: 75) is greater than (Operator) a threshold (Value 2: 70)
- Set Yes Value to "Turn off AC" and No Value to "Keep AC running"
- The block would output "Turn off AC" if the temperature is above 70, and "Keep AC running" if it's below 70

