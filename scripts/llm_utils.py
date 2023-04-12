from typing import List, Dict, Optional, Any

import openai
from config import Config
import logging
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt

cfg = Config()

openai.api_key = cfg.openai_api_key


class ChatRequest(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    message_list: List[Dict[str, str]]
    model: str = cfg.fast_llm_model
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    deployment_id: Optional[str] = cfg.openai_deployment_id
    key: Optional[Any] = None

    @classmethod
    def from_messages(cls, messages: List[Dict[str, str]] | str, model: str = cfg.fast_llm_model,
                      system_role: Optional[str] = None, key: Optional[Any] = None,
                      deployment_id: Optional[str] = cfg.openai_deployment_id, max_tokens: Optional[int] = None,
                      temperature: float = 0.0):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if system_role:
            messages.insert(0, {
                'role': 'system',
                'content': system_role
            })
        return ChatRequest(message_list=messages, model=model, max_tokens=max_tokens, temperature=temperature,
                           deployment_id=deployment_id, key=key)


class ChatResponse(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    request: ChatRequest
    reply: Optional[str]
    exception: Optional[BaseException] = None

    def key(self) -> Optional[Any]:
        return self.request.key


def create_chat_completion(messages, model=None, temperature=None, max_tokens=None) -> str:
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
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

    return response.choices[0].message["content"]


async def async_chat_completion(request: ChatRequest, return_exceptions: bool = True) -> ChatResponse:
    try:
        logging.debug(f"creating completion for {request.key or request.message_list[-1]['content'][-50:]}")
        return await _async_chat_completion(request)
    except Exception as e:
        if return_exceptions:
            return ChatResponse(request=request, reply=None, exception=e)
        else:
            logging.exception(f'async_chat_completion {e}')
            raise e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def _async_chat_completion(request: ChatRequest) -> ChatResponse:
    if request.deployment_id:
        response = openai.ChatCompletion.create(
            deployment_id=request.deployment_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=request.model,
            messages=request.message_list,
        )
    else:
        response = openai.ChatCompletion.create(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=request.model,
            messages=request.message_list,
        )
    content = dict(list(dict(response.items())["choices"])[0])["message"]["content"]
    return ChatResponse(request=request, reply=content)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
