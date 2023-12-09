import typing

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .sdk.forge_log import ForgeLogger
from litellm import completion, acompletion, AuthenticationError, InvalidRequestError

LOG = ForgeLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(
    model, messages, **kwargs
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        kwargs["model"] = model
        kwargs["messages"] = messages

        resp = await acompletion(**kwargs)
        return resp
    except AuthenticationError as e:
        LOG.exception("Authentication Error")
    except InvalidRequestError as e:
        LOG.exception("Invalid Request Error")
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_embedding_request(
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
