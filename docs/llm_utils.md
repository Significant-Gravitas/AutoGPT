## Function `create_chat_completion`

```python
import openai
from config import Config

def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    """Create a chat completion using the OpenAI API"""
```

This is a function that creates a chatbot response using OpenAI API. It takes in the following arguments:
- `messages` (required): A list of conversation history examples to use as context for the chatbot's response.
- `model` (optional): The name of the model to use for generating the response. If not provided, the default OpenAI model will be used.
- `temperature` (optional): The sampling temperature to use when generating the response. Higher values means the chatbot may be more creative or unpredictable, while lower values will stick closer to learned patterns.
- `max_tokens` (optional): The maximum number of tokens in the response. If not provided, a default maximum token length will be used.

### Example Usage
```python
cfg = Config()
openai.api_key = cfg.openai_api_key

# Example conversation history
messages = ["Hello there!", "How are you doing today?"]

# Generate a chatbot response
response = create_chat_completion(messages)

print(response)
```

#### Output
```
I'm doing alright, thanks for asking. How can I assist you today?
```