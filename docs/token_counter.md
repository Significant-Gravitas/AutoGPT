# `count_message_tokens` and `count_string_tokens` functions

This module provides two functions for counting the number of tokens in a message or string. The `count_message_tokens` function takes a list of messages and a model name, and returns the total number of tokens in all the messages. The `count_string_tokens` function takes a string and a model name, and returns the total number of tokens in the string.

## `count_message_tokens` function

```python
def count_message_tokens(
    messages: List[Message], model: str = "gpt-3.5-turbo-0301"
) -> int:
```

### Arguments
- `messages` : List[Message]
    - A list of messages, each of which is a dictionary containing the role and content of the message.
- `model` : str = "gpt-3.5-turbo-0301"
    - The name of the model to use for tokenization. Defaults to "gpt-3.5-turbo-0301".

### Returns
- `int` : The number of tokens used by the list of messages.

### Example
```python
from autogpt.utils.token_counter import count_message_tokens
messages = [
    {
        "role": "user",
        "content": "Hi, how are you doing today?",
        "name": None,
    },
    {
        "role": "assistant",
        "content": "I'm doing pretty well, thanks. How can I assist you?",
        "name": None,
    },
]
print(count_message_tokens(messages)) # Output : 31
```

## `count_string_tokens` function
```python
def count_string_tokens(string: str, model_name: str) -> int:
```

### Arguments
- `string` : str 
    - The text string.
- `model_name` : str
    - The name of the encoding to use (e.g., "gpt-3.5-turbo").

### Returns
- `int` : The number of tokens in the text string.

### Example
```python
from autogpt.utils.token_counter import count_string_tokens
string = "This is some sample text."
model_name = "gpt-3.5-turbo-0301"
print(count_string_tokens(string, model_name)) # Output : 6
```