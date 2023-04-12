## Class `Singleton`
### Methods
- `_instances`

This is a metaclass for ensuring that only one instance of a class exists.

## Class `AbstractSingleton`
This is an abstract class for the singleton metaclass.

## Class `Config`
### Methods
- `__init__(self)`
    - Initializes the `Config` class.

- `set_continuous_mode(self, value: bool)`
    - set the continuous mode value.

- `set_speak_mode(self, value: bool)`
    - set the speak mode value.

- `set_fast_llm_model(self, value: str)`
    - set the fast LLM model value.

- `set_smart_llm_model(self, value: str)`
    - set the smart LLM model value.

- `set_fast_token_limit(self, value: int)`
    - set the fast token limit value.

- `set_smart_token_limit(self, value: int)`
    - set the smart token limit value.

- `set_openai_api_key(self, value: str)`
    - set the OpenAI API key value.

- `set_elevenlabs_api_key(self, value: str)`
    - set the ElevenLabs API key value.

- `set_google_api_key(self, value: str)`
    - set the Google API key value.

- `set_custom_search_engine_id(self, value: str)`
    - set the custom search engine id value.

- `set_pinecone_api_key(self, value: str)`
    - set the Pinecone API key value.

- `set_pinecone_region(self, value: str)`
    - set the Pinecone region value.

- `set_debug_mode(self, value: bool)`
    - set the debug mode value.

This is a configuration class that stores the state of bools for different scripts access. It also sets and gets various external API keys and IDs for external services like Azure, Google, etc.