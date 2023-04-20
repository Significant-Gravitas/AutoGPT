# Configuration Class

This is a singleton configuration class that stores the state of bools for different scripts access. It is used to set attributes that can be accessed globally. It has default values that can be overriden by .env files or other configuration files.

## Methods

### `__init__(self)`

Initializes the `Config` class, sets default values for various attributes.

### `get_azure_deployment_id_for_model(self, model: str) -> str`

Returns the relevant deployment id for the model specified.

__Args__:
- `model(str)`: The model to map to the deployment id.

__Returns__:
- The matching deployment id if found, otherwise an empty string.

### `load_azure_config(self, config_file: str = AZURE_CONFIG_FILE) -> None`

Loads the configuration parameters for Azure hosting from the specified file path as a yaml file.

__Args__:
- `config_file(str)`: The path to the config yaml file. DEFAULT: "../azure.yaml".

__Returns__:
- None.

### `set_continuous_mode(self, value: bool) -> None`

Sets the continuous mode value.

__Args__:
- `value(bool)`: The value to assign to the `continuous_mode` attribute.

__Returns__:
- None.

### `set_continuous_limit(self, value: int) -> None`

Sets the continuous limit value.

__Args__:
- `value(int)`: The value to assign to the `continuous_limit` attribute.

__Returns__:
- None.

### `set_speak_mode(self, value: bool) -> None`

Sets the speak mode value.

__Args__:
- `value(bool)`: The value to assign to the `speak_mode` attribute.

__Returns__:
- None.

### `set_fast_llm_model(self, value: str) -> None`

Sets the fast LLM model value.

__Args__:
- `value(str)`: The value to assign to the `fast_llm_model` attribute.

__Returns__:
- None.

### `set_smart_llm_model(self, value: str) -> None`

Sets the smart LLM model value.

__Args__:
- `value(str)`: The value to assign to the `smart_llm_model` attribute.

__Returns__:
- None.

### `set_fast_token_limit(self, value: int) -> None`

Sets the fast token limit value.

__Args__:
- `value(int)`: The value to assign to the `fast_token_limit` attribute.

__Returns__:
- None.

### `set_smart_token_limit(self, value: int) -> None`

Sets the smart token limit value.

__Args__:
- `value(int)`: The value to assign to the `smart_token_limit` attribute.

__Returns__:
- None.

### `set_browse_chunk_max_length(self, value: int) -> None`

Sets the browse_website command chunk max length value.

__Args__:
- `value(int)`: The value to assign to the `browse_chunk_max_length` attribute.

__Returns__:
- None.

### `set_openai_api_key(self, value: str) -> None`

Sets the OpenAI API key value.

__Args__:
- `value(str)`: The value to assign to the `openai_api_key` attribute.

__Returns__:
- None.

### `set_elevenlabs_api_key(self, value: str) -> None`

Sets the ElevenLabs API key value.

__Args__:
- `value(str)`: The value to assign to the `elevenlabs_api_key` attribute.

__Returns__:
- None.

### `set_elevenlabs_voice_1_id(self, value: str) -> None`

Sets the ElevenLabs Voice 1 ID value.

__Args__:
- `value(str)`: The value to assign to the `elevenlabs_voice_1_id` attribute.

__Returns__:
- None.

### `set_elevenlabs_voice_2_id(self, value: str) -> None`

Sets the ElevenLabs Voice 2 ID value.

__Args__:
- `value(str)`: The value to assign to the `elevenlabs_voice_2_id` attribute.

__Returns__:
- None.

### `set_google_api_key(self, value: str) -> None`

Sets the Google API key value.

__Args__:
- `value(str)`: The value to assign to the `google_api_key` attribute.

__Returns__:
- None.

### `set_custom_search_engine_id(self, value: str) -> None`

Sets the custom search engine id value.

__Args__:
- `value(str)`: The value to assign to the `custom_search_engine_id` attribute.

__Returns__:
- None.

### `set_pinecone_api_key(self, value: str) -> None`

Sets the Pinecone API key value.

__Args__:
- `value(str)`: The value to assign to the `pinecone_api_key` attribute.

__Returns__:
- None.

### `set_pinecone_region(self, value: str) -> None`

Sets the Pinecone region value.

__Args__:
- `value(str)`: The value to assign to the `pinecone_region` attribute.

__Returns__:
- None.

### `set_debug_mode(self, value: bool) -> None`

Sets the debug mode value.

__Args__:
- `value(bool)`: The value to assign to the `debug_mode` attribute.

__Returns__:
- None.

### `set_plugins(self, value: list) -> None`

Sets the plugins value.

__Args__:
- `value(list)`: The value to assign to the `plugins` attribute.

__Returns__:
- None.

### `check_openai_api_key() -> None`

Checks if the OpenAI API key is set in config.py or as an environment variable. If not set, prints an error message with instructions for setting the key and exits the program. 

__Args__:
- None.

__Returns__:
- None.