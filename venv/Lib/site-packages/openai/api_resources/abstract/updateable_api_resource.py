from urllib.parse import quote_plus
from typing import Awaitable

from openai.api_resources.abstract.api_resource import APIResource


class UpdateableAPIResource(APIResource):
    @classmethod
    def modify(cls, sid, **params):
        url = "%s/%s" % (cls.class_url(), quote_plus(sid))
        return cls._static_request("post", url, **params)

    @classmethod
    def amodify(cls, sid, **params) -> Awaitable:
        url = "%s/%s" % (cls.class_url(), quote_plus(sid))
        return cls._astatic_request("patch", url, **params)
