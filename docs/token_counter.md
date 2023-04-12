### `count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0301") -> int`

Counts the number of tokens used by a list of messages.

**Args:**
- `messages` (List[Dict[str, str]]): A list of messages, each of which is a dictionary containing the role and content of the message.
- `model` (str): The name of the model to use for tokenization. Defaults to "gpt-3.5-turbo-0301".

**Returns:**
- `int`: The number of tokens used by the list of messages.

**Example:**
```python
>>> messages = [{'role':'user', 'content':'Hello'}, {'role':'assistant', 'content':'Hi there! How may I assist you?'}]
>>> count_message_tokens(messages)
19
```

### `count_string_tokens(string: str, model_name: str) -> int`

Counts the number of tokens in a text string.

**Args:**
- `string` (str): The text string.
- `model_name` (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

**Returns:**
- `int`: The number of tokens in the text string.

**Example:**
```python
>>> count_string_tokens("Hello! How are you doing today?", "gpt-3.5-turbo")
10
```