# Base OpenAI Plugin

This code defines a plugin called BaseOpenAIPlugin which is used for generating Auto-GPT plugins. This class extends `AutoGPTPluginTemplate` and provides a base for more specific plugin classes. This module also imports `typing` module and creates a `TypedDict` for `Message`.

## Classes

### BaseOpenAIPlugin

This class is for generating Auto-GPT plugins. It implements several methods for handling various aspects of a chatbot interaction such as responses, planning, instructions, commands and chat completion etc. The implemented methods can be customized with subclassing.

#### Methods

- `__init__(self, manifests_specs_clients: dict)`: The constructor initializes the plugin with the required parameters. The required parameters are:
  * `manifests_specs_clients`: Dictionary containing manifest, specs, and client.
  
- `can_handle_on_response(self) -> bool`: A method that checks whether the plugin can handle the `on_response` method or not. Returns a boolean value of True if the plugin can handle the `on_response` method and False otherwise.
  
- `on_response(self, response: str, *args, **kwargs) -> str`: A method that is called when a response is received from the model.
  * `response`: the response string
  * `args` and `kwargs`: extra arguments
  
- `can_handle_post_prompt(self) -> bool`: A method that checks whether the plugin can handle the `post_prompt` method or not. Returns a boolean value of True if the plugin can handle the `post_prompt` method and False otherwise.
  
- `post_prompt(self, prompt: PromptGenerator) -> PromptGenerator`: A method that is called just after the generate_prompt is called, but actually before the prompt is generated.
  * `prompt`: the prompt generator.
  
- `can_handle_on_planning(self) -> bool`: A method that checks whether the plugin can handle the `on_planning` method or not. Returns a boolean value of True if the plugin can handle the `on_planning` method and False otherwise.
  
- `on_planning(self, prompt: PromptGenerator, messages: List[Message]) -> Optional[str]`: A method that is called before the planning chat completion is done.
  * `prompt`: the prompt generator
  * `messages`: the list of messages
  
- `can_handle_post_planning(self) -> bool`: A method that checks whether the plugin can handle the `post_planning` method or not. Returns a boolean value of True if the plugin can handle the `post_planning` method and False otherwise.
  
- `post_planning(self, response: str) -> str`: A method that is called after the planning chat completion is done.
  * `response`: The response string.
  
- `can_handle_pre_instruction(self) -> bool`: A method that checks whether the plugin can handle the `pre_instruction` method or not. Returns a boolean value of True if the plugin can handle the `pre_instruction` method and False otherwise.
  
- `pre_instruction(self, messages: List[Message]) -> List[Message]`: A method that is called before the instruction chat is done. Returns the resulting list of messages.
  * `messages`: The list of context messages.
  
- `can_handle_on_instruction(self) -> bool`: A method that checks whether the plugin can handle the `on_instruction` method or not. Returns a boolean value of True if the plugin can handle the `on_instruction` method and False otherwise.
  
- `on_instruction(self, messages: List[Message]) -> Optional[str]`: A method that is called when the instruction chat is done. Returns the resulting message.
  * `messages`: The list of context messages.
  
- `can_handle_post_instruction(self) -> bool`: A method that checks whether the plugin can handle the `post_instruction` method or not. Returns a boolean value of True if the plugin can handle the `post_instruction` method and False otherwise.
  
- `post_instruction(self, response: str) -> str`: A method that is called after the instruction chat is done. Returns the resulting response.
  * `response`: The response string.
  
- `can_handle_pre_command(self) -> bool`: A method that checks whether the plugin can handle the `pre_command` method or not. Returns a boolean value of True if the plugin can handle the `pre_command` method and False otherwise.
  
- `pre_command(self, command_name: str, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]`: A method that is called before the command is executed. Returns the command name and the arguments.
  * `command_name`: The command name.
  * `arguments`: The arguments.
  
- `can_handle_post_command(self) -> bool`: A method that checks whether the plugin can handle the `post_command` method or not. Returns a boolean value of True if the plugin can handle the `post_command` method and False otherwise.
  
- `post_command(self, command_name: str, response: str) -> str`: A method that is called after the command is executed. Returns the resulting response.
  * `command_name`: The command name.
  * `response`: The response string.
  
- `can_handle_chat_completion(self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int) -> bool`: A method that checks whether the plugin can handle the `chat_completion` method or not. Returns a boolean value of True if the plugin can handle the `chat_completion` method and False otherwise.
  * `messages`: The messages.
  * `model`: The model name.
  * `temperature`: The temperature.
  * `max_tokens`: The max tokens.

- `handle_chat_completion(self, messages: List[Message], model: str, temperature: float, max_tokens: int) -> str`: A method that is called when the chat completion is done. Returns the resulting response.
  * `messages`: The messages.
  * `model`: The model name.
  * `temperature`: The temperature.
  * `max_tokens`: The max tokens.