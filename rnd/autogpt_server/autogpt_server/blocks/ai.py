import logging
from enum import Enum

import openai

from autogpt_server.data.block import Block, BlockOutput, BlockSchema, BlockFieldSecret
from autogpt_server.util import json

logger = logging.getLogger(__name__)


class LlmModel(str, Enum):
    openai_gpt4 = "gpt-4-turbo"


class LlmCallBlock(Block):
    class Input(BlockSchema):
        prompt: str
        api_key: BlockFieldSecret = BlockFieldSecret(key="openai_api_key")
        sys_prompt: str = ""
        expected_format: dict[str, str] = {}
        model: LlmModel = LlmModel.openai_gpt4
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
                "model": "gpt-4-turbo",
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
    def llm_call(api_key: str, model: LlmModel, prompt: list[dict], json: bool) -> str:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            messages=prompt,  # type: ignore
            response_format={"type": "json_object"} if json else None,
        )
        return response.choices[0].message.content or ""

    def run(self, input_data: Input) -> BlockOutput:
        prompt = []

        def trim_prompt(s: str) -> str:
            lines = s.strip().split("\n")
            return "\n".join([line.strip().lstrip("|") for line in lines])

        if input_data.sys_prompt:
            prompt.append({"role": "system", "content": input_data.sys_prompt})

        if input_data.expected_format:
            expected_format = [f'"{k}": "{v}"' for k, v in
                               input_data.expected_format.items()]

            format_prompt = ",\n  ".join(expected_format)
            sys_prompt = f"""
              |Reply in json format:
              |{{
              |  {format_prompt}                
              |}}
            """
            prompt.append({"role": "system", "content": trim_prompt(sys_prompt)})

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
        for retry_count in range(input_data.retry):
            response_text = self.llm_call(
                api_key=input_data.api_key.get(),
                model=input_data.model,
                prompt=prompt,
                json=bool(input_data.expected_format),
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

            retry_prompt = f"""
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
            prompt.append({"role": "user", "content": trim_prompt(retry_prompt)})

        yield "error", retry_prompt
