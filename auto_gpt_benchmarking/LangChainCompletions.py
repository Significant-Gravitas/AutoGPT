import importlib
from typing import Optional
from evals.api import CompletionFn, CompletionResult

from langchain.llms import BaseLLM

from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling


class LangChainLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class LangChainLLMCompletionFn(CompletionFn):
    def __init__(self, llm: str, llm_kwargs: Optional[dict] = {}, **kwargs) -> None:
        # Import and resolve self.llm to an instance of llm argument here, assuming it's always a subclass of BaseLLM
        module = importlib.import_module("langchain.llms")
        LLMClass = getattr(module, llm)

        if issubclass(LLMClass, BaseLLM):
            self.llm = LLMClass(**llm_kwargs)
        else:
            raise ValueError(f"{llm} is not a subclass of BaseLLM")

    def __call__(self, prompt, **kwargs) -> LangChainLLMCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.llm(prompt)
        record_sampling(prompt=prompt, sampled=response)
        return LangChainLLMCompletionResult(response)
