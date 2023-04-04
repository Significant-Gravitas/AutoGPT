import time
from pydoc import apropos
from typing import Optional
from urllib.parse import quote_plus

import openai
from openai import api_requestor, error, util
from openai.api_resources.abstract.api_resource import APIResource
from openai.openai_response import OpenAIResponse
from openai.util import ApiType

MAX_TIMEOUT = 20


class EngineAPIResource(APIResource):
    plain_old_data = False

    def __init__(self, engine: Optional[str] = None, **kwargs):
        super().__init__(engine=engine, **kwargs)

    @classmethod
    def class_url(
        cls,
        engine: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        # Namespaces are separated in object names with periods (.) and in URLs
        # with forward slashes (/), so replace the former with the latter.
        base = cls.OBJECT_NAME.replace(".", "/")  # type: ignore
        typed_api_type, api_version = cls._get_api_type_and_version(
            api_type, api_version
        )

        if typed_api_type in (ApiType.AZURE, ApiType.AZURE_AD):
            if not api_version:
                raise error.InvalidRequestError(
                    "An API version is required for the Azure API type."
                )
            if engine is None:
                raise error.InvalidRequestError(
                    "You must provide the deployment name in the 'engine' parameter to access the Azure OpenAI service"
                )
            extn = quote_plus(engine)
            return "/%s/%s/%s/%s?api-version=%s" % (
                cls.azure_api_prefix,
                cls.azure_deployments_prefix,
                extn,
                base,
                api_version,
            )

        elif typed_api_type == ApiType.OPEN_AI:
            if engine is None:
                return "/%s" % (base)

            extn = quote_plus(engine)
            return "/engines/%s/%s" % (extn, base)

        else:
            raise error.InvalidAPIType("Unsupported API type %s" % api_type)

    @classmethod
    def __prepare_create_request(
        cls,
        api_key=None,
        api_base=None,
        api_type=None,
        api_version=None,
        organization=None,
        **params,
    ):
        deployment_id = params.pop("deployment_id", None)
        engine = params.pop("engine", deployment_id)
        model = params.get("model", None)
        timeout = params.pop("timeout", None)
        stream = params.get("stream", False)
        headers = params.pop("headers", None)
        request_timeout = params.pop("request_timeout", None)
        typed_api_type = cls._get_api_type_and_version(api_type=api_type)[0]
        if typed_api_type in (util.ApiType.AZURE, util.ApiType.AZURE_AD):
            if deployment_id is None and engine is None:
                raise error.InvalidRequestError(
                    "Must provide an 'engine' or 'deployment_id' parameter to create a %s"
                    % cls,
                    "engine",
                )
        else:
            if model is None and engine is None:
                raise error.InvalidRequestError(
                    "Must provide an 'engine' or 'model' parameter to create a %s"
                    % cls,
                    "engine",
                )

        if timeout is None:
            # No special timeout handling
            pass
        elif timeout > 0:
            # API only supports timeouts up to MAX_TIMEOUT
            params["timeout"] = min(timeout, MAX_TIMEOUT)
            timeout = (timeout - params["timeout"]) or None
        elif timeout == 0:
            params["timeout"] = MAX_TIMEOUT

        requestor = api_requestor.APIRequestor(
            api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            organization=organization,
        )
        url = cls.class_url(engine, api_type, api_version)
        return (
            deployment_id,
            engine,
            timeout,
            stream,
            headers,
            request_timeout,
            typed_api_type,
            requestor,
            url,
            params,
        )

    @classmethod
    def create(
        cls,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        (
            deployment_id,
            engine,
            timeout,
            stream,
            headers,
            request_timeout,
            typed_api_type,
            requestor,
            url,
            params,
        ) = cls.__prepare_create_request(
            api_key, api_base, api_type, api_version, organization, **params
        )

        response, _, api_key = requestor.request(
            "post",
            url,
            params=params,
            headers=headers,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            # must be an iterator
            assert not isinstance(response, OpenAIResponse)
            return (
                util.convert_to_openai_object(
                    line,
                    api_key,
                    api_version,
                    organization,
                    engine=engine,
                    plain_old_data=cls.plain_old_data,
                )
                for line in response
            )
        else:
            obj = util.convert_to_openai_object(
                response,
                api_key,
                api_version,
                organization,
                engine=engine,
                plain_old_data=cls.plain_old_data,
            )

            if timeout is not None:
                obj.wait(timeout=timeout or None)

        return obj

    @classmethod
    async def acreate(
        cls,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        (
            deployment_id,
            engine,
            timeout,
            stream,
            headers,
            request_timeout,
            typed_api_type,
            requestor,
            url,
            params,
        ) = cls.__prepare_create_request(
            api_key, api_base, api_type, api_version, organization, **params
        )
        response, _, api_key = await requestor.arequest(
            "post",
            url,
            params=params,
            headers=headers,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            # must be an iterator
            assert not isinstance(response, OpenAIResponse)
            return (
                util.convert_to_openai_object(
                    line,
                    api_key,
                    api_version,
                    organization,
                    engine=engine,
                    plain_old_data=cls.plain_old_data,
                )
                async for line in response
            )
        else:
            obj = util.convert_to_openai_object(
                response,
                api_key,
                api_version,
                organization,
                engine=engine,
                plain_old_data=cls.plain_old_data,
            )

            if timeout is not None:
                await obj.await_(timeout=timeout or None)

        return obj

    def instance_url(self):
        id = self.get("id")

        if not isinstance(id, str):
            raise error.InvalidRequestError(
                f"Could not determine which URL to request: {type(self).__name__} instance has invalid ID: {id}, {type(id)}. ID should be of type str.",
                "id",
            )

        extn = quote_plus(id)
        params_connector = "?"

        if self.typed_api_type in (ApiType.AZURE, ApiType.AZURE_AD):
            api_version = self.api_version or openai.api_version
            if not api_version:
                raise error.InvalidRequestError(
                    "An API version is required for the Azure API type."
                )
            base = self.OBJECT_NAME.replace(".", "/")
            url = "/%s/%s/%s/%s/%s?api-version=%s" % (
                self.azure_api_prefix,
                self.azure_deployments_prefix,
                self.engine,
                base,
                extn,
                api_version,
            )
            params_connector = "&"

        elif self.typed_api_type == ApiType.OPEN_AI:
            base = self.class_url(self.engine, self.api_type, self.api_version)
            url = "%s/%s" % (base, extn)

        else:
            raise error.InvalidAPIType("Unsupported API type %s" % self.api_type)

        timeout = self.get("timeout")
        if timeout is not None:
            timeout = quote_plus(str(timeout))
            url += params_connector + "timeout={}".format(timeout)
        return url

    def wait(self, timeout=None):
        start = time.time()
        while self.status != "complete":
            self.timeout = (
                min(timeout + start - time.time(), MAX_TIMEOUT)
                if timeout is not None
                else MAX_TIMEOUT
            )
            if self.timeout < 0:
                del self.timeout
                break
            self.refresh()
        return self

    async def await_(self, timeout=None):
        """Async version of `EngineApiResource.wait`"""
        start = time.time()
        while self.status != "complete":
            self.timeout = (
                min(timeout + start - time.time(), MAX_TIMEOUT)
                if timeout is not None
                else MAX_TIMEOUT
            )
            if self.timeout < 0:
                del self.timeout
                break
            await self.arefresh()
        return self
