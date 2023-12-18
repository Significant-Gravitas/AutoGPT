import enum
import math
import os
import tiktoken
from typing import Any, Callable, Dict, ParamSpec, Tuple, TypeVar, ClassVar

from openai import AsyncOpenAI, completions
from openai.resources import AsyncCompletions, AsyncEmbeddings


aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

from AFAAS.configs import (Configurable,
                                                         SystemConfiguration,
                                                         UserConfigurable)
from AFAAS.interfaces.adapters.chatmodel import (CompletionModelFunction)
from AFAAS.interfaces.adapters.language_model import (
    BaseModelProviderBudget, BaseModelProviderConfiguration, BaseModelProviderCredentials,
    BaseModelProviderSettings, BaseModelProviderUsage, Embedding,
    EmbeddingModelInfo, EmbeddingModelProvider, EmbeddingModelResponse,
    ModelProviderName, ModelProviderService, ModelTokenizer)
from AFAAS.core.adapters.openai.common import _OpenAIRetryHandler, OPEN_AI_CHAT_MODELS, OPEN_AI_EMBEDDING_MODELS, OPEN_AI_MODELS, OpenAIModelName, OpenAISettings
from AFAAS.lib.utils.json_schema import JSONSchema

from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class EmbeddingsOpenAI(EmbeddingModelProvider):
    """A provider for OpenAI's API.

    Provides methods to communicate with OpenAI's API and generate responses.

    Attributes:
        default_settings: The default settings for the OpenAI provider.
    """


    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using the OpenAI API.

        Args:
            text (str): The text to embed.
            model_name (str): The name of the embedding model.
            embedding_parser (Callable): A parser to process the embedding.
            **kwargs: Additional keyword arguments.

        Returns:
            EmbeddingModelResponse: Response containing the embedding.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> embedding_response = await provider.create_embedding("Hello, world!", ...)
            >>> print(embedding_response.embedding)
            [0.123, -0.456, ...]
        """
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response_args = {
            "model_info": OPEN_AI_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response = EmbeddingModelResponse(
            **response_args,
            embedding=embedding_parser(response.embeddings[0]),
        )
        self._budget.update_usage_and_cost(response)
        return response


    def _get_embedding_kwargs(
        self,
        model_name: OpenAIModelName,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            dict: Dictionary containing the kwargs.

        Example:
            >>> provider = OpenAIProvider(...)
            >>> embedding_kwargs = provider._get_embedding_kwargs(OpenAIModelName.ADA, ...)
            >>> print(embedding_kwargs)
            {'model': 'text-embedding-ada-002', ...}
        """
        embedding_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.unmasked(),
        }

        return embedding_kwargs


async def _create_embedding(text: str, *_, **kwargs) -> AsyncEmbeddings:
    """Embed text using the OpenAI API.

    Args:
        text str: The text to embed.
        model_name str: The name of the model to use.

    Returns:
        str: The embedding.
    """
    return await aclient.embeddings.create(input=[text], **kwargs)
