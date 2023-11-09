from typing import Optional

from openai.openai_object import OpenAIObject
from openai.util import merge_dicts


class ErrorObject(OpenAIObject):
    def refresh_from(
        self,
        values,
        api_key=None,
        api_version=None,
        api_type=None,
        organization=None,
        response_ms: Optional[int] = None,
    ):
        # Unlike most other API resources, the API will omit attributes in
        # error objects when they have a null value. We manually set default
        # values here to facilitate generic error handling.
        values = merge_dicts({"message": None, "type": None}, values)
        return super(ErrorObject, self).refresh_from(
            values=values,
            api_key=api_key,
            api_version=api_version,
            api_type=api_type,
            organization=organization,
            response_ms=response_ms,
        )
