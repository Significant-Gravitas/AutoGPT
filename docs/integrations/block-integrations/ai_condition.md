# AI Condition Block

## What it is
The AI Condition Block is a logical component that uses artificial intelligence to evaluate natural language conditions and produces outputs based on the result. This block allows you to define conditions in plain English rather than using traditional comparison operators.

## What it does
This block takes an input value and a natural language condition, then uses AI to determine whether the input satisfies the condition. Based on the result, it provides conditional outputs similar to a traditional if/else statement but with the flexibility of natural language evaluation.

## How it works
The block uses a Large Language Model (LLM) to evaluate the condition by:
1. Converting the input value to a string representation
2. Sending a carefully crafted prompt to the AI asking it to evaluate whether the input meets the specified condition
3. Parsing the AI's response to determine a true/false result
4. Outputting the appropriate value based on the result

## Inputs
| Input | Description |
|-------|-------------|
| Input Value | The value to be evaluated (can be text, number, or any data type) |
| Condition | A plaintext English description of the condition to evaluate |
| Yes Value | (Optional) The value to output if the condition is true. If not provided, Input Value will be used |
| No Value | (Optional) The value to output if the condition is false. If not provided, Input Value will be used |
| Model | The LLM model to use for evaluation (defaults to GPT-4o) |
| Credentials | API credentials for the LLM provider |

## Outputs
| Output | Description |
|--------|-------------|
| Result | A boolean value (true or false) indicating whether the condition was met |
| Yes Output | The output value if the condition is true. This will be the Yes Value if provided, or Input Value if not |
| No Output | The output value if the condition is false. This will be the No Value if provided, or Input Value if not |
| Error Message | Error message if the AI evaluation is uncertain or fails (empty string if successful) |

## Examples

### Email Address Validation
- **Input Value**: `"john@example.com"`
- **Condition**: `"the input is an email address"`
- **Result**: `true`
- **Yes Output**: `"john@example.com"` (or custom Yes Value)

### Geographic Location Check
- **Input Value**: `"San Francisco"`
- **Condition**: `"the input is a city in the USA"`
- **Result**: `true`
- **Yes Output**: `"San Francisco"` (or custom Yes Value)

### Error Detection
- **Input Value**: `"Error: Connection timeout"`
- **Condition**: `"the input is an error message or refusal"`
- **Result**: `true`
- **Yes Output**: `"Error: Connection timeout"` (or custom Yes Value)

### Content Classification
- **Input Value**: `"This is a detailed explanation of how machine learning works..."`
- **Condition**: `"the input is the body of an email"`
- **Result**: `false` (it's more like article content)
- **No Output**: Custom No Value or the input value

## Possible Use Cases
- **Content Classification**: Automatically classify text content (emails, articles, comments, etc.)
- **Data Validation**: Validate input data using natural language rules
- **Smart Routing**: Route data through different paths based on AI-evaluated conditions
- **Quality Control**: Check if content meets certain quality or format standards
- **Language Detection**: Determine if text is in a specific language or style
- **Sentiment Analysis**: Evaluate if content has positive, negative, or neutral sentiment
- **Error Handling**: Detect and route error messages or problematic inputs

## Advantages over Traditional Condition Blocks
- **Flexibility**: Can handle complex, nuanced conditions that would be difficult to express with simple comparisons
- **Natural Language**: Uses everyday language instead of programming logic
- **Context Awareness**: AI can understand context and meaning, not just exact matches
- **Adaptability**: Can handle variations in input format and wording

## Considerations
- **Performance**: Requires an API call to an LLM, which adds latency compared to traditional conditions
- **Cost**: Each evaluation consumes LLM tokens, which has associated costs
- **Reliability**: AI responses may occasionally be inconsistent, so critical logic should include fallback handling
- **Network Dependency**: Requires internet connectivity to access the LLM API