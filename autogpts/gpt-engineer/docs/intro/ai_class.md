# AI Class Documentation
The AI class is the main interface to the GPT-4 model. It provides methods to start a conversation with the model, continue an existing conversation, and format system and user messages. The AI class is defined in the `gpt_engineer/ai.py` file.

<br>

## Methods
`__init__(self, model="gpt-4", temperature=0.1)`: The constructor takes the name of the AI model and the temperature parameter as arguments. It tries to retrieve the specified model from OpenAI. If the model is not available, it reverts to a fallback model.

`start(self, system, user)`: This method starts a conversation with the AI. It takes a system message and a user message as arguments, and returns the AI's response.

`fsystem(self, msg)`: This method formats a system message. It takes a message as an argument and returns a dictionary with the role set to "system" and the content set to the message.

`fuser(self, msg)`: This method formats a user message. It takes a message as an argument and returns a dictionary with the role set to "user" and the content set to the message.

`fassistant(self, msg)`: This method formats an assistant message. It takes a message as an argument and returns a dictionary with the role set to "assistant" and the content set to the message.

`next(self, messages: list[dict[str, str]], prompt=None)`: This method continues a conversation with the AI. It takes a list of messages and an optional prompt as arguments, and returns the AI's response.
