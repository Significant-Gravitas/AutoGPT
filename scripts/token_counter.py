import tiktoken
from typing import List, Dict


def count_message_tokens(messages : List[Dict[str, str]], model : str = "gpt-3.5-turbo-0301") -> int:
    """
    返回由消息列表使用的标记数量。

    参数：
    messages（list）：消息列表，每个消息都是包含消息角色和内容的字典。
    model（str）：用于标记化的模型名称。默认为"gpt-3.5-turbo-0301"。

    返回：
    int：消息列表使用的标记数量。
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warn("Warning：未找到模型。使用 cl100k_base 编码。")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # !Node: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return count_message_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # !Note: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return count_message_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_string_tokens(string: str, model_name: str) -> int:
    """
    返回文本字符串中的标记数量。

    参数：
    string（str）：文本字符串。
    model_name（str）：要使用的编码名称（例如，“gpt-3.5-turbo”）。

    返回：
    int：文本字符串中的标记数量。
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
