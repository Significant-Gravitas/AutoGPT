from urllib.parse import quote_plus

from openai import api_requestor, util, error
from openai.api_resources.abstract import (
    CreateableAPIResource,
    ListableAPIResource,
    nested_resource_class_methods,
)
from openai.api_resources.abstract.deletable_api_resource import DeletableAPIResource
from openai.openai_response import OpenAIResponse
from openai.util import ApiType


@nested_resource_class_methods("event", operations=["list"])
class FineTune(ListableAPIResource, CreateableAPIResource, DeletableAPIResource):
    OBJECT_NAME = "fine-tunes"

    @classmethod
    def _prepare_cancel(
        cls,
        id,
        api_key=None,
        api_type=None,
        request_id=None,
        api_version=None,
        **params,
    ):
        base = cls.class_url()
        extn = quote_plus(id)

        typed_api_type, api_version = cls._get_api_type_and_version(
            api_type, api_version
        )
        if typed_api_type in (ApiType.AZURE, ApiType.AZURE_AD):
            url = "/%s%s/%s/cancel?api-version=%s" % (
                cls.azure_api_prefix,
                base,
                extn,
                api_version,
            )
        elif typed_api_type == ApiType.OPEN_AI:
            url = "%s/%s/cancel" % (base, extn)
        else:
            raise error.InvalidAPIType("Unsupported API type %s" % api_type)

        instance = cls(id, api_key, **params)
        return instance, url

    @classmethod
    def cancel(
        cls,
        id,
        api_key=None,
        api_type=None,
        request_id=None,
        api_version=None,
        **params,
    ):
        instance, url = cls._prepare_cancel(
            id,
            api_key,
            api_type,
            request_id,
            api_version,
            **params,
        )
        return instance.request("post", url, request_id=request_id)

    @classmethod
    def acancel(
        cls,
        id,
        api_key=None,
        api_type=None,
        request_id=None,
        api_version=None,
        **params,
    ):
        instance, url = cls._prepare_cancel(
            id,
            api_key,
            api_type,
            request_id,
            api_version,
            **params,
        )
        return instance.arequest("post", url, request_id=request_id)

    @classmethod
    def _prepare_stream_events(
        cls,
        id,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        base = cls.class_url()
        extn = quote_plus(id)

        requestor = api_requestor.APIRequestor(
            api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            organization=organization,
        )

        typed_api_type, api_version = cls._get_api_type_and_version(
            api_type, api_version
        )

        if typed_api_type in (ApiType.AZURE, ApiType.AZURE_AD):
            url = "/%s%s/%s/events?stream=true&api-version=%s" % (
                cls.azure_api_prefix,
                base,
                extn,
                api_version,
            )
        elif typed_api_type == ApiType.OPEN_AI:
            url = "%s/%s/events?stream=true" % (base, extn)
        else:
            raise error.InvalidAPIType("Unsupported API type %s" % api_type)

        return requestor, url

    @classmethod
    def stream_events(
        cls,
        id,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        requestor, url = cls._prepare_stream_events(
            id,
            api_key,
            api_base,
            api_type,
            request_id,
            api_version,
            organization,
            **params,
        )

        response, _, api_key = requestor.request(
            "get", url, params, stream=True, request_id=request_id
        )

        assert not isinstance(response, OpenAIResponse)  # must be an iterator
        return (
            util.convert_to_openai_object(
                line,
                api_key,
                api_version,
                organization,
            )
            for line in response
        )

    @classmethod
    async def astream_events(
        cls,
        id,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        requestor, url = cls._prepare_stream_events(
            id,
            api_key,
            api_base,
            api_type,
            request_id,
            api_version,
            organization,
            **params,
        )

        response, _, api_key = await requestor.arequest(
            "get", url, params, stream=True, request_id=request_id
        )

        assert not isinstance(response, OpenAIResponse)  # must be an iterator
        return (
            util.convert_to_openai_object(
                line,
                api_key,
                api_version,
                organization,
            )
            async for line in response
        )
