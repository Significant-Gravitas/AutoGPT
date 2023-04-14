import time
import openai
from colorama import Fore
from config import Config

cfg = Config()

openai.api_key = cfg.openai_api_key


# Overly simple abstraction until we create something better
# simple retry mechanism when getting a rate error or a bad gateway
def create_chat_completion(messages, model=None, temperature=cfg.temperature, max_tokens=None)->str:
    """使用OpenAI API创建聊天完成"""
    response = None
    num_retries = 5
    for attempt in range(num_retries):
        try:
            if cfg.use_azure:
                response = openai.ChatCompletion.create(
                    deployment_id=cfg.get_azure_deployment_id_for_model(model),
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            break
        except openai.error.RateLimitError:
            if cfg.debug_mode:
                print(Fore.RED + "Error: ", "已达到 API 速率限制。 等待 20 秒......" + Fore.RESET)
            time.sleep(20)
        except openai.error.APIError as e:
            if e.http_status == 502:
                if cfg.debug_mode:
                    print(Fore.RED + "Error: ", "API 网关错误。 等待 20 秒..." + Fore.RESET)
                time.sleep(20)
            else:
                raise
            if attempt == num_retries - 1:
                raise

    if response is None:
        raise RuntimeError("5 次重试后未能得到响应")

    return response.choices[0].message["content"]
