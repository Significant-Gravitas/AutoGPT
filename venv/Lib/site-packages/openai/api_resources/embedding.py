import base64
import time


from openai import util
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from openai.datalib import numpy as np, assert_has_numpy
from openai.error import TryAgain


class Embedding(EngineAPIResource):
    OBJECT_NAME = "embeddings"

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creates a new embedding for the provided input and parameters.

        See https://platform.openai.com/docs/api-reference/embeddings for a list
        of valid parameters.
        """
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        user_provided_encoding_format = kwargs.get("encoding_format", None)

        # If encoding format was not explicitly specified, we opaquely use base64 for performance
        if not user_provided_encoding_format:
            kwargs["encoding_format"] = "base64"

        while True:
            try:
                response = super().create(*args, **kwargs)

                # If a user specifies base64, we'll just return the encoded string.
                # This is only for the default case.
                if not user_provided_encoding_format:
                    for data in response.data:

                        # If an engine isn't using this optimization, don't do anything
                        if type(data["embedding"]) == str:
                            assert_has_numpy()
                            data["embedding"] = np.frombuffer(
                                base64.b64decode(data["embedding"]), dtype="float32"
                            ).tolist()

                return response
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        Creates a new embedding for the provided input and parameters.

        See https://platform.openai.com/docs/api-reference/embeddings for a list
        of valid parameters.
        """
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        user_provided_encoding_format = kwargs.get("encoding_format", None)

        # If encoding format was not explicitly specified, we opaquely use base64 for performance
        if not user_provided_encoding_format:
            kwargs["encoding_format"] = "base64"

        while True:
            try:
                response = await super().acreate(*args, **kwargs)

                # If a user specifies base64, we'll just return the encoded string.
                # This is only for the default case.
                if not user_provided_encoding_format:
                    for data in response.data:

                        # If an engine isn't using this optimization, don't do anything
                        if type(data["embedding"]) == str:
                            data["embedding"] = np.frombuffer(
                                base64.b64decode(data["embedding"]), dtype="float32"
                            ).tolist()

                return response
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)
