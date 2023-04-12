### Functionality
This module contains functions to interact with the OpenAI API, specifically to generate chat conversations between a user and an AI. The `chat_with_ai` function is the main function that takes in inputs from the user, generates context based on the history of messages, and sends the context to the model for generation. 

### Required Libraries
The following libraries are required for this module to work
- `time`
- `openai`
- `dotenv`
- `config`
- `token_counter` 
- `llm_utils`

### `create_chat_message` function
Function to create a chat message with the given role and content.
##### Inputs 
- `role`: A string, the role of the message sender, e.g., "system", "user", or "assistant".
- `content`: A string, the content of the message.

##### Output 
- A dictionary containing the role and content of the message.

### `generate_context` function
Function to generate context based on the prompt, relevant memory, full history of messages and model. 
##### Inputs
- `prompt`: A string, prompt explaining the rules to the AI.
- `relevant_memory`: A string, contains relevant past message history.
- `full_message_history`: A list that stores all messages sent between the user and the AI.
- `model`: A string, containing the name of pre-trained model.

##### Output 
- `next_message_to_add_index`:  An integer representing the next message to add index.
- `current_tokens_used`: An integer representing currently used tokens.
- `insertion_index`: An integer representing insertion index .
- `current_context`: A list containing the current context.

### `chat_with_ai` function
Main function to interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory.
##### Inputs
- `prompt`: A string, the prompt explaining the rules to the AI.
- `user_input`: A string containing the input from the user.
- `full_message_history`: A list that stores all messages sent between the user and the AI.
- `permanent_memory`: A memory object containing the permanent memory.
- `token_limit`: An integer, maximum number of tokens allowed in the API call.

##### Output
- A string, the AI's response.