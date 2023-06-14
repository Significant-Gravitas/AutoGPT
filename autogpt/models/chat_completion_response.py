class ChatCompletionResponse:
    def __init__(self, content: str, function_call: dict[str, str]):
        self.content = content
        self.function_call = function_call
