import typing
import os

import openai
import google.generativeai as palm
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .forge_log import ForgeLogger

LOG = ForgeLogger(__name__)

########################
#    OpenAI LLM        #
########################

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(
    messages,
    functions=None,
    function_call=None,
    model=str,
    custom_labels=None,
    temperature=None) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        kwargs = {
            "model": model,
            "messages": messages,
        }

        if functions:
            kwargs["functions"] = functions

        if function_call:
            kwargs["function_call"] = function_call

        if custom_labels:
            kwargs["headers"] = {}
            for label in custom_labels.keys():
                # This is an example showing adding in the labels as helicone properties
                kwargs["headers"][f"Helicone-Property-{label}"] = custom_labels[label]
        
        # tuning temperature
        if temperature:
            kwargs["temperature"] = temperature
        
        resp = await openai.ChatCompletion.acreate(**kwargs)

        return resp
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_chat_embedding_request(
    messages, model="text-embedding-ada-002"
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        return await openai.Embedding.acreate(
            input=[f"{m['role']}: {m['content']}" for m in messages],
            engine=model,
        )
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_text_embedding_request(
    text, model="text-embedding-ada-002"
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        return await openai.Embedding.acreate(
            input=[text],
            engine=model,
        )
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def transcribe_audio(
    audio_file: str,
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Transcribe an audio file using OpenAI's API"""
    try:
        return await openai.Audio.transcribe(model="whisper-1", file=audio_file)
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

########################
#    Google LLM        #
########################
# msg format
# author = role
# { "author": "...", "content": "..." }
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def palm_chat_request(messages: str | typing.List, temperature: float = 1.0) -> str | None:
    palm.configure(api_key=os.getenv("PALM_API_KEY"))

    resp = await palm.chat_async(
        messages=messages,
        temperature=temperature
    )
    
    return resp.last