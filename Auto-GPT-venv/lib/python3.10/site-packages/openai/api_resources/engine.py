import time
import warnings

from openai import util
from openai.api_resources.abstract import ListableAPIResource, UpdateableAPIResource
from openai.error import TryAgain


class Engine(ListableAPIResource, UpdateableAPIResource):
    OBJECT_NAME = "engines"

    def generate(self, timeout=None, **params):
        start = time.time()
        while True:
            try:
                return self.request(
                    "post",
                    self.instance_url() + "/generate",
                    params,
                    stream=params.get("stream"),
                    plain_old_data=True,
                )
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)

    async def agenerate(self, timeout=None, **params):
        start = time.time()
        while True:
            try:
                return await self.arequest(
                    "post",
                    self.instance_url() + "/generate",
                    params,
                    stream=params.get("stream"),
                    plain_old_data=True,
                )
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)

    def embeddings(self, **params):
        warnings.warn(
            "Engine.embeddings is deprecated, use Embedding.create", DeprecationWarning
        )
        return self.request("post", self.instance_url() + "/embeddings", params)
