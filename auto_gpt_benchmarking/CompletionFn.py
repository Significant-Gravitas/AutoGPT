from evals.api import CompletionFn, CompletionResult

from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling
from auto_gpt_benchmarking.AutoGPTAgent import AutoGPTAgent


class AutoGPTCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class AutoGPTCompletionFn(CompletionFn):

    def __init__(self, auto_gpt_path, **kwargs) -> None:
        self.auto_gpt_path = auto_gpt_path
        self.agent = None

    def __call__(self, prompt, **kwargs) -> AutoGPTCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        self.kill_agent()
        self.agent = AutoGPTAgent(prompt, self.auto_gpt_path)
        response = self.agent.start()
        record_sampling(prompt=prompt, sampled=response)
        return AutoGPTCompletionResult(response)

    def kill_agent(self):
        if self.agent:
            self.agent.kill()


