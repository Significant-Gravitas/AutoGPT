import os

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

class Singleton(type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self):
        self.continuous_mode = False
        self.speak_mode = False
        # TODO - make these models be self-contained, using langchain, so we can configure them once and call it good 
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo") 
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-4")
        self.thinking_token_limit = 4000
        # Initialize the OpenAI API client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        # Print values:
        print("Config values:")
        print(f"continuous_mode: {self.continuous_mode}")
        print(f"speak_mode: {self.speak_mode}")
        print(f"fast_llm_model: {self.fast_llm_model}")
        print(f"smart_llm_model: {self.smart_llm_model}")
        print(f"thinking_token_limit: {self.thinking_token_limit}")
        print(f"openai_api_key: {self.openai_api_key}")
        print(f"elevenlabs_api_key: {self.elevenlabs_api_key}")

        # Initialize the OpenAI API client
        openai.api_key = self.openai_api_key


    def set_continuous_mode(self, value: bool):
        self.continuous_mode = value

    def set_speak_mode(self, value: bool):
        self.speak_mode = value

    def set_fast_llm_model(self, value: str):
        self.fast_llm_model = value

    def set_smart_llm_model(self, value: str):
        self.smart_llm_model = value

    def set_thinking_token_limit(self, value: int):
        self.thinking_token_limit = value

    def set_openai_api_key(self, value: str):
        self.apiopenai_api_key_key = value
    
    def set_elevenlabs_api_key(self, value: str):
        self.elevenlabs_api_key = value


        
