import logging
from enum import Enum
from json import JSONDecodeError
from typing import Any, List, NamedTuple

import anthropic
import ollama
import openai
from groq import Groq

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import BlockSecret, SchemaField, SecretField
from autogpt_server.util import json

logger = logging.getLogger(__name__)

LlmApiKeys = {
    "openai": BlockSecret("openai_api_key"),
    "anthropic": BlockSecret("anthropic_api_key"),
    "groq": BlockSecret("groq_api_key"),
    "ollama": BlockSecret(value=""),
}


class ModelMetadata(NamedTuple):
    provider: str
    context_window: int
    cost_factor: int


class LlmModel(str, Enum):
    # OpenAI models
    GPT4O_MINI = "gpt-4o-mini"
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
    # New Groq models (Preview)
    LLAMA3_1_405B = "llama-3.1-405b-reasoning"
    LLAMA3_1_70B = "llama-3.1-70b-versatile"
    LLAMA3_1_8B = "llama-3.1-8b-instant"
    # Ollama models
    OLLAMA_LLAMA3_8B = "llama3"
    OLLAMA_LLAMA3_405B = "llama3.1:405b"

    @property
    def metadata(self) -> ModelMetadata:
        return MODEL_METADATA[self]


MODEL_METADATA = {
    LlmModel.GPT4O_MINI: ModelMetadata("openai", 128000, cost_factor=10),
    LlmModel.GPT4O: ModelMetadata("openai", 128000, cost_factor=12),
    LlmModel.GPT4_TURBO: ModelMetadata("openai", 128000, cost_factor=11),
    LlmModel.GPT3_5_TURBO: ModelMetadata("openai", 16385, cost_factor=8),
    LlmModel.CLAUDE_3_5_SONNET: ModelMetadata("anthropic", 200000, cost_factor=14),
    LlmModel.CLAUDE_3_HAIKU: ModelMetadata("anthropic", 200000, cost_factor=13),
    LlmModel.LLAMA3_8B: ModelMetadata("groq", 8192, cost_factor=6),
    LlmModel.LLAMA3_70B: ModelMetadata("groq", 8192, cost_factor=9),
    LlmModel.MIXTRAL_8X7B: ModelMetadata("groq", 32768, cost_factor=7),
    LlmModel.GEMMA_7B: ModelMetadata("groq", 8192, cost_factor=6),
    LlmModel.GEMMA2_9B: ModelMetadata("groq", 8192, cost_factor=7),
    LlmModel.LLAMA3_1_405B: ModelMetadata("groq", 8192, cost_factor=10),
    # Limited to 16k during preview
    LlmModel.LLAMA3_1_70B: ModelMetadata("groq", 131072, cost_factor=15),
    LlmModel.LLAMA3_1_8B: ModelMetadata("groq", 131072, cost_factor=13),
    LlmModel.OLLAMA_LLAMA3_8B: ModelMetadata("ollama", 8192, cost_factor=7),
    LlmModel.OLLAMA_LLAMA3_405B: ModelMetadata("ollama", 8192, cost_factor=11),
}

for model in LlmModel:
    if model not in MODEL_METADATA:
        raise ValueError(f"Missing MODEL_METADATA metadata for model: {model}")


class AIStructuredResponseGeneratorBlock(Block):
    class Input(BlockSchema):
        prompt: str
        expected_format: dict[str, str]
        model: LlmModel = LlmModel.GPT4_TURBO
        api_key: BlockSecret = SecretField(value="")
        sys_prompt: str = ""
        retry: int = 3
        prompt_values: dict[str, str] = SchemaField(
            advanced=False, default={}, description="Values used to fill in the prompt."
        )

    class Output(BlockSchema):
        response: dict[str, Any]
        error: str

    def __init__(self):
        super().__init__(
            id="ed55ac19-356e-4243-a6cb-bc599e9b716f",
            description="Call a Large Language Model (LLM) to generate formatted object based on the given prompt.",
            categories={BlockCategory.AI},
            input_schema=AIStructuredResponseGeneratorBlock.Input,
            output_schema=AIStructuredResponseGeneratorBlock.Output,
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
            test_mock={
                "llm_call": lambda *args, **kwargs: json.dumps(
                    {
                        "key1": "key1Value",
                        "key2": "key2Value",
                    }
                )
            },
        )

    @staticmethod
    def llm_call(
        api_key: str, model: LlmModel, prompt: list[dict], json_format: bool
    ) -> str:
        provider = model.metadata.provider

        if provider == "openai":
            openai.api_key = api_key
            response_format = {"type": "json_object"} if json_format else None
            response = openai.chat.completions.create(
                model=model.value,
                messages=prompt,  # type: ignore
                response_format=response_format,  # type: ignore
            )
            return response.choices[0].message.content or ""
        elif provider == "anthropic":
            system_messages = [p["content"] for p in prompt if p["role"] == "system"]
            sysprompt = " ".join(system_messages)

            messages = []
            last_role = None
            for p in prompt:
                if p["role"] in ["user", "assistant"]:
                    if p["role"] != last_role:
                        messages.append({"role": p["role"], "content": p["content"]})
                        last_role = p["role"]
                    else:
                        # If the role is the same as the last one, combine the content
                        messages[-1]["content"] += "\n" + p["content"]

            client = anthropic.Anthropic(api_key=api_key)
            try:
                response = client.messages.create(
                    model=model.value,
                    max_tokens=4096,
                    system=sysprompt,
                    messages=messages,
                )
                return response.content[0].text if response.content else ""
            except anthropic.APIError as e:
                error_message = f"Anthropic API error: {str(e)}"
                logger.error(error_message)
                raise ValueError(error_message)
        elif provider == "groq":
            client = Groq(api_key=api_key)
            response_format = {"type": "json_object"} if json_format else None
            response = client.chat.completions.create(
                model=model.value,
                messages=prompt,  # type: ignore
                response_format=response_format,  # type: ignore
            )
            return response.choices[0].message.content or ""
        elif provider == "ollama":
            response = ollama.generate(
                model=model.value,
                prompt=prompt[0]["content"],
            )
            return response["response"]
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def run(self, input_data: Input) -> BlockOutput:
        prompt = []

        def trim_prompt(s: str) -> str:
            lines = s.strip().split("\n")
            return "\n".join([line.strip().lstrip("|") for line in lines])

        values = input_data.prompt_values
        if values:
            input_data.prompt = input_data.prompt.format(**values)
            input_data.sys_prompt = input_data.sys_prompt.format(**values)

        if input_data.sys_prompt:
            prompt.append({"role": "system", "content": input_data.sys_prompt})

        if input_data.expected_format:
            expected_format = [
                f'"{k}": "{v}"' for k, v in input_data.expected_format.items()
            ]
            format_prompt = ",\n  ".join(expected_format)
            sys_prompt = trim_prompt(
                f"""
              |Reply in json format:
              |{{
              |  {format_prompt}
              |}}
            """
            )
            prompt.append({"role": "system", "content": sys_prompt})

        prompt.append({"role": "user", "content": input_data.prompt})

        def parse_response(resp: str) -> tuple[dict[str, Any], str | None]:
            try:
                parsed = json.loads(resp)
                if not isinstance(parsed, dict):
                    return {}, f"Expected a dictionary, but got {type(parsed)}"
                miss_keys = set(input_data.expected_format.keys()) - set(parsed.keys())
                if miss_keys:
                    return parsed, f"Missing keys: {miss_keys}"
                return parsed, None
            except JSONDecodeError as e:
                return {}, f"JSON decode error: {e}"

        logger.info(f"LLM request: {prompt}")
        retry_prompt = ""
        model = input_data.model
        api_key = (
            input_data.api_key.get_secret_value()
            or LlmApiKeys[model.metadata.provider].get_secret_value()
        )

        for retry_count in range(input_data.retry):
            try:
                response_text = self.llm_call(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    json_format=bool(input_data.expected_format),
                )
                logger.info(f"LLM attempt-{retry_count} response: {response_text}")

                if input_data.expected_format:
                    parsed_dict, parsed_error = parse_response(response_text)
                    if not parsed_error:
                        yield "response", {
                            k: (
                                json.loads(v)
                                if isinstance(v, str)
                                and v.startswith("[")
                                and v.endswith("]")
                                else (", ".join(v) if isinstance(v, list) else v)
                            )
                            for k, v in parsed_dict.items()
                        }
                        return
                else:
                    yield "response", {"response": response_text}
                    return

                retry_prompt = trim_prompt(
                    f"""
                  |This is your previous error response:
                  |--
                  |{response_text}
                  |--
                  |
                  |And this is the error:
                  |--
                  |{parsed_error}
                  |--
                """
                )
                prompt.append({"role": "user", "content": retry_prompt})
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                retry_prompt = f"Error calling LLM: {e}"

        yield "error", retry_prompt


class AITextGeneratorBlock(Block):
    class Input(BlockSchema):
        prompt: str
        model: LlmModel = LlmModel.GPT4_TURBO
        api_key: BlockSecret = SecretField(value="")
        sys_prompt: str = ""
        retry: int = 3
        prompt_values: dict[str, str] = SchemaField(
            advanced=False, default={}, description="Values used to fill in the prompt."
        )

    class Output(BlockSchema):
        response: str
        error: str

    def __init__(self):
        super().__init__(
            id="1f292d4a-41a4-4977-9684-7c8d560b9f91",
            description="Call a Large Language Model (LLM) to generate a string based on the given prompt.",
            categories={BlockCategory.AI},
            input_schema=AITextGeneratorBlock.Input,
            output_schema=AITextGeneratorBlock.Output,
            test_input={"prompt": "User prompt"},
            test_output=("response", "Response text"),
            test_mock={"llm_call": lambda *args, **kwargs: "Response text"},
        )

    @staticmethod
    def llm_call(input_data: AIStructuredResponseGeneratorBlock.Input) -> str:
        object_block = AIStructuredResponseGeneratorBlock()
        for output_name, output_data in object_block.run(input_data):
            if output_name == "response":
                return output_data["response"]
            else:
                raise output_data
        raise ValueError("Failed to get a response from the LLM.")

    def run(self, input_data: Input) -> BlockOutput:
        try:
            object_input_data = AIStructuredResponseGeneratorBlock.Input(
                **{attr: getattr(input_data, attr) for attr in input_data.model_fields},
                expected_format={},
            )
            yield "response", self.llm_call(object_input_data)
        except Exception as e:
            yield "error", str(e)


class AITextSummarizerBlock(Block):
    class Input(BlockSchema):
        text: str
        model: LlmModel = LlmModel.GPT4_TURBO
        api_key: BlockSecret = SecretField(value="")
        # TODO: Make this dynamic
        max_tokens: int = 4000  # Adjust based on the model's context window
        chunk_overlap: int = 100  # Overlap between chunks to maintain context

    class Output(BlockSchema):
        summary: str
        error: str

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-7g8h-9i0j-1k2l-m3n4o5p6q7r8",
            description="Utilize a Large Language Model (LLM) to summarize a long text.",
            categories={BlockCategory.AI, BlockCategory.TEXT},
            input_schema=AITextSummarizerBlock.Input,
            output_schema=AITextSummarizerBlock.Output,
            test_input={"text": "Lorem ipsum..." * 100},
            test_output=("summary", "Final summary of a long text"),
            test_mock={
                "llm_call": lambda input_data: (
                    {"final_summary": "Final summary of a long text"}
                    if "final_summary" in input_data.expected_format
                    else {"summary": "Summary of a chunk of text"}
                )
            },
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            for output in self._run(input_data):
                yield output
        except Exception as e:
            yield "error", str(e)

    def _run(self, input_data: Input) -> BlockOutput:
        chunks = self._split_text(
            input_data.text, input_data.max_tokens, input_data.chunk_overlap
        )
        summaries = []

        for chunk in chunks:
            chunk_summary = self._summarize_chunk(chunk, input_data)
            summaries.append(chunk_summary)

        final_summary = self._combine_summaries(summaries, input_data)
        yield "summary", final_summary

    @staticmethod
    def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
        words = text.split()
        chunks = []
        chunk_size = max_tokens - overlap

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + max_tokens])
            chunks.append(chunk)

        return chunks

    @staticmethod
    def llm_call(
        input_data: AIStructuredResponseGeneratorBlock.Input,
    ) -> dict[str, str]:
        llm_block = AIStructuredResponseGeneratorBlock()
        for output_name, output_data in llm_block.run(input_data):
            if output_name == "response":
                return output_data
        raise ValueError("Failed to get a response from the LLM.")

    def _summarize_chunk(self, chunk: str, input_data: Input) -> str:
        prompt = f"Summarize the following text concisely:\n\n{chunk}"

        llm_response = self.llm_call(
            AIStructuredResponseGeneratorBlock.Input(
                prompt=prompt,
                api_key=input_data.api_key,
                model=input_data.model,
                expected_format={"summary": "The summary of the given text."},
            )
        )

        return llm_response["summary"]

    def _combine_summaries(self, summaries: list[str], input_data: Input) -> str:
        combined_text = " ".join(summaries)

        if len(combined_text.split()) <= input_data.max_tokens:
            prompt = (
                "Provide a final, concise summary of the following summaries:\n\n"
                + combined_text
            )

            llm_response = self.llm_call(
                AIStructuredResponseGeneratorBlock.Input(
                    prompt=prompt,
                    api_key=input_data.api_key,
                    model=input_data.model,
                    expected_format={
                        "final_summary": "The final summary of all provided summaries."
                    },
                )
            )

            return llm_response["final_summary"]
        else:
            # If combined summaries are still too long, recursively summarize
            return self._run(
                AITextSummarizerBlock.Input(
                    text=combined_text,
                    api_key=input_data.api_key,
                    model=input_data.model,
                    max_tokens=input_data.max_tokens,
                    chunk_overlap=input_data.chunk_overlap,
                )
            ).send(None)[
                1
            ]  # Get the first yielded value


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BlockSchema):
    role: MessageRole
    content: str


class AIConversationBlock(Block):
    class Input(BlockSchema):
        messages: List[Message] = SchemaField(
            description="List of messages in the conversation.", min_length=1
        )
        model: LlmModel = SchemaField(
            default=LlmModel.GPT4_TURBO,
            description="The language model to use for the conversation.",
        )
        api_key: BlockSecret = SecretField(
            value="", description="API key for the chosen language model provider."
        )
        max_tokens: int | None = SchemaField(
            default=None,
            description="The maximum number of tokens to generate in the chat completion.",
            ge=1,
        )

    class Output(BlockSchema):
        response: str = SchemaField(
            description="The model's response to the conversation."
        )
        error: str = SchemaField(description="Error message if the API call failed.")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-g7h8-i9j0-k1l2-m3n4o5p6q7r8",
            description="Advanced LLM call that takes a list of messages and sends them to the language model.",
            categories={BlockCategory.AI},
            input_schema=AIConversationBlock.Input,
            output_schema=AIConversationBlock.Output,
            test_input={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {
                        "role": "assistant",
                        "content": "The Los Angeles Dodgers won the World Series in 2020.",
                    },
                    {"role": "user", "content": "Where was it played?"},
                ],
                "model": LlmModel.GPT4_TURBO,
                "api_key": "test_api_key",
            },
            test_output=(
                "response",
                "The 2020 World Series was played at Globe Life Field in Arlington, Texas.",
            ),
            test_mock={
                "llm_call": lambda *args, **kwargs: "The 2020 World Series was played at Globe Life Field in Arlington, Texas."
            },
        )

    @staticmethod
    def llm_call(
        api_key: str,
        model: LlmModel,
        messages: List[dict[str, str]],
        max_tokens: int | None = None,
    ) -> str:
        provider = model.metadata.provider

        if provider == "openai":
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model=model.value,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        elif provider == "anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model.value,
                max_tokens=max_tokens or 4096,
                messages=messages,  # type: ignore
            )
            return response.content[0].text if response.content else ""
        elif provider == "groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model.value,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        elif provider == "ollama":
            response = ollama.chat(
                model=model.value,
                messages=messages,  # type: ignore
                stream=False,  # type: ignore
            )
            return response["message"]["content"]
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def run(self, input_data: Input) -> BlockOutput:
        try:
            api_key = (
                input_data.api_key.get_secret_value()
                or LlmApiKeys[input_data.model.metadata.provider].get_secret_value()
            )

            messages = [message.model_dump() for message in input_data.messages]

            response = self.llm_call(
                api_key=api_key,
                model=input_data.model,
                messages=messages,
                max_tokens=input_data.max_tokens,
            )

            yield "response", response
        except Exception as e:
            yield "error", f"Error calling LLM: {str(e)}"
