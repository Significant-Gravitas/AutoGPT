# Condition Block

## What it is
The Condition Block is a logical component that evaluates comparisons between two values and produces outputs based on the result.

## What it does
This block compares two input values using a specified comparison operator and determines whether the condition is true or false. It then outputs the result of the comparison and provides corresponding output values for both true and false cases.

## How it works
The block takes two values and a comparison operator as inputs. It then performs the comparison using the specified operator. Based on the result of the comparison, it outputs a boolean value (true or false) and the corresponding output value for the true or false case.

## Inputs
| Input | Description |
|-------|-------------|
| Value 1 | The first value to be compared. This can be any type of value (number, text, or boolean) |
| Operator | The comparison operator to use (e.g., equal to, not equal to, greater than, less than) |
| Value 2 | The second value to be compared. This can be any type of value (number, text, or boolean) |
| Yes Value | (Optional) The value to output if the condition is true. If not provided, Value 1 will be used |
| No Value | (Optional) The value to output if the condition is false. If not provided, Value 1 will be used |

## Outputs
| Output | Description |
|--------|-------------|
| Result | A boolean value (true or false) indicating whether the condition was met |
| Yes Output | The output value if the condition is true. This will be the Yes Value if provided, or Value 1 if not |
| No Output | The output value if the condition is false. This will be the No Value if provided, or Value 1 if not |

## Possible use case
This block could be used in a customer loyalty program to determine if a customer qualifies for a discount. For example, you could compare the customer's total purchases (Value 1) with a threshold amount (Value 2) using the "greater than or equal to" operator. The Yes Value could be "Qualified for discount" and the No Value could be "Not qualified". The block would then output whether the customer qualifies and the appropriate message.