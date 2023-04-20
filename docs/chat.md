# `chat_with_ai`


## Description
This function facilitates interaction between a user and an AI model. It prompts the user for input, sends it to the AI model for processing, and returns its response. The conversation history is updated with each interaction.

## Signature
```python
def chat_with_ai(
    agent, prompt, user_input, full_message_history, permanent_memory, token_limit
)
```

## Parameters
* `agent` (type: object): An object that provides methods for communicating and receiving data between the chat interface and the AI model.
* `prompt` (type: str): The prompt to be displayed to the user.
* `user_input` (type: str): The user's input.
* `full_message_history` (type: list): A list of messages sent between the user and AI. Each message is represented as a dictionary containing the role and content of the sender.
* `permanent_memory` (type: object): An object containing the permanent memory.
* `token_limit` (type: int): The maximum number of tokens allowed in the API call.

## Returns
* Returns string: The AI's response.

## Example
```python
agent = "agent"
prompt = "Hi, how can I assist you?"
user_input = "Can you tell me about the benefits of exercise?"
full_message_history = []
permanent_memory = "permanent_memory"
token_limit = 100

chat_with_ai(agent, prompt, user_input, full_message_history, permanent_memory, token_limit)
```