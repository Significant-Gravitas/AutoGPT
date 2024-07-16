import logging
from enum import Enum
from typing import NamedTuple

import openai
import anthropic
from groq import Groq

from autogpt_server.data.block import Block, BlockOutput, BlockSchema, BlockFieldSecret
from autogpt_server.util import json

logger = logging.getLogger(__name__)

LlmApiKeys = {
    "openai": BlockFieldSecret("openai_api_key"),
    "anthropic": BlockFieldSecret("anthropic_api_key"),
    "groq": BlockFieldSecret("groq_api_key"),
}


class ModelMetadata(NamedTuple):
    provider: str
    context_window: int


class LlmModel(str, Enum):
    # OpenAI models
    GPT4O = "gpt-4o"
    GPT4_TURBO = "gpt-4-turbo"
    GPT3_5_TURBO = "gpt-3.5-turbo"
    # Anthropic models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    # Groq models
    LLAMA3_8B = "llama3-8b-8192"
    LLAMA3_70B = "llama3-70b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"
    GEMMA2_9B = "gemma2-9b-it"

    @property
    def metadata(self) -> ModelMetadata:
        return MODEL_METADATA[self]


MODEL_METADATA = {
    LlmModel.GPT4O: ModelMetadata("openai", 128000),
    LlmModel.GPT4_TURBO: ModelMetadata("openai", 128000),
    LlmModel.GPT3_5_TURBO: ModelMetadata("openai", 16385),
    LlmModel.CLAUDE_3_5_SONNET: ModelMetadata("anthropic", 200000),
    LlmModel.CLAUDE_3_HAIKU: ModelMetadata("anthropic", 200000),
    LlmModel.LLAMA3_8B: ModelMetadata("groq", 8192),
    LlmModel.LLAMA3_70B: ModelMetadata("groq", 8192),
    LlmModel.MIXTRAL_8X7B: ModelMetadata("groq", 32768),
    LlmModel.GEMMA_7B: ModelMetadata("groq", 8192),
    LlmModel.GEMMA2_9B: ModelMetadata("groq", 8192),
}


class LlmCallBlock(Block):
    class Input(BlockSchema):
        prompt: str
        model: LlmModel = LlmModel.GPT4_TURBO
        api_key: BlockFieldSecret = BlockFieldSecret(value="")
        sys_prompt: str = ""
        expected_format: dict[str, str] = {}
        retry: int = 3

    class Output(BlockSchema):
        response: dict[str, str]
        error: str

    def __init__(self):
        super().__init__(
            id="ed55ac19-356e-4243-a6cb-bc599e9b716f",
            input_schema=LlmCallBlock.Input,
            output_schema=LlmCallBlock.Output,
            test_input={
                "model": LlmModel.GPT4_TURBO,
                "api_key": "fake-api",
                "expected_format": {
                    "key1": "value1",
                    "key2": "value2",
                },
                "prompt": "User prompt",
            },
            test_output=("response", {"key1": "key1Value", "key2": "key2Value"}),
            test_mock={"llm_call": lambda *args, **kwargs: json.dumps({
                "key1": "key1Value",
                "key2": "key2Value",
            })},
        )

    @staticmethod
    def llm_call(api_key: str, model: LlmModel, prompt: list[dict], json_format: bool) -> str:
        provider = model.metadata.provider
        
        if provider == "openai":
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model=model.value,
                messages=prompt,  # type: ignore
                response_format={"type": "json_object"} if json_format else None, # type: ignore
            )
            return response.choices[0].message.content or ""
        elif provider == "anthropic":
            sysprompt = "".join([p["content"] for p in prompt if p["role"] == "system"])
            usrprompt = [p for p in prompt if p["role"] == "user"]
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model.value,
                max_tokens=4096,
                system=sysprompt,
                messages=usrprompt,  # type: ignore
            )
            return response.content[0].text if response.content else ""
        elif provider == "groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model.value,
                messages=prompt,  # type: ignore
                response_format={"type": "json_object"} if json_format else None,
            )
            return response.choices[0].message.content or ""
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def run(self, input_data: Input) -> BlockOutput:
        prompt = []

        def trim_prompt(s: str) -> str:
            lines = s.strip().split("\n")
            return "\n".join([line.strip().lstrip("|") for line in lines])

        if input_data.sys_prompt:
            prompt.append({"role": "system", "content": input_data.sys_prompt})

        if input_data.expected_format:
            expected_format = [f'"{k}": "{v}"' for k, v in input_data.expected_format.items()]
            format_prompt = ",\n  ".join(expected_format)
            sys_prompt = trim_prompt(f"""
              |Reply in json format:
              |{{
              |  {format_prompt}                
              |}}
            """)
            prompt.append({"role": "system", "content": sys_prompt})

        prompt.append({"role": "user", "content": input_data.prompt})

        def parse_response(resp: str) -> tuple[dict[str, str], str | None]:
            try:
                parsed = json.loads(resp)
                miss_keys = set(input_data.expected_format.keys()) - set(parsed.keys())
                if miss_keys:
                    return parsed, f"Missing keys: {miss_keys}"
                return parsed, None
            except Exception as e:
                return {}, f"JSON decode error: {e}"

        logger.warning(f"LLM request: {prompt}")
        retry_prompt = ""
        model = input_data.model
        api_key = input_data.api_key.get() or LlmApiKeys[model.metadata.provider].get()

        for retry_count in range(input_data.retry):
            try:
                response_text = self.llm_call(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    json_format=bool(input_data.expected_format),
                )
                logger.warning(f"LLM attempt-{retry_count} response: {response_text}")

                if input_data.expected_format:
                    parsed_dict, parsed_error = parse_response(response_text)
                    if not parsed_error:
                        yield "response", {k: str(v) for k, v in parsed_dict.items()}
                        return
                else:
                    yield "response", {"response": response_text}
                    return

                retry_prompt = trim_prompt(f"""
                  |This is your previous error response:
                  |--
                  |{response_text}
                  |--
                  |
                  |And this is the error:
                  |--
                  |{parsed_error}
                  |--
                """)
                prompt.append({"role": "user", "content": retry_prompt})
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                retry_prompt = f"Error calling LLM: {e}"

        yield "error", retry_prompt
