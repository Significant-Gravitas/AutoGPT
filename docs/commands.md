# AI Agent Controller

This module contains functions that allow users to control AI agents and perform various actions such as memory modification, file operations, and web browsing. 

## Dependencies

* `browse`
* `json`
* `memory`
* `datetime`
* `agent_manager`
* `speak`
* `config`
* `ai_functions`
* `file_operations`
* `execute_code`
* `json_parser`
* `image_gen`
* `duckduckgo_search`
* `googleapiclient`

## Functions

### `is_valid_int(value) -> bool`

This function returns `True` if the input value is a valid integer and `False` otherwise.

### `get_command(response) -> Tuple[str, str]`

This function takes in a JSON-formatted string and returns a tuple containing the name of the command and its arguments. If the response is not in valid JSON format or is missing a required field, an error message is returned.

### `execute_command(command_name, arguments) -> str`

This function takes in the name of the command and its arguments and executes the command. If the command is not recognized or encounters an error, an error message is returned.

### `get_datetime() -> str`

This function returns the current date and time.

### `google_search(query, num_results=8) -> str`

This function takes in a search query and returns a list of URLs based on the query. Uses an unofficial search method if the Google API key is not set, and the official Google API search method if the key is set.

### `google_official_search(query, num_results=8) -> str`

This function takes in a search query and returns a list of URLs based on the query using the official Google API search method.

### `browse_website(url, question) -> str`

This function takes in a URL and a question, then browses the website and returns a summary of its content and a list of links. 

### `get_text_summary(url, question) -> str`

This function takes in a URL and a question, then returns a summary of the website's content.

### `get_hyperlinks(url) -> str`

This function takes in a URL and returns a list of links on the website.

### `commit_memory(string) -> str`

This function takes in a string and commits it to permanent memory.

### `delete_memory(key) -> str`

This function deletes the memory at a given key.

### `overwrite_memory(key, string) -> str`

This function overwrites the memory at a given key with a new string.

### `shutdown() -> str`

This function shuts down the program.

### `start_agent(name, task, prompt, model=cfg.fast_llm_model) -> str`

This function creates a new agent with the given name, task, prompt, and model.

### `message_agent(key, message) -> str`

This function sends a message to an agent with the given key and returns the agent's response.

### `list_agents() -> str`

This function returns a list of all agents.

### `delete_agent(key) -> str`

This function deletes an agent with the given key.