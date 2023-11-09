from urllib.parse import quote_plus

import openai
from openai import api_requestor, error, util
from openai.openai_object import OpenAIObject
from openai.util import ApiType
from typing import Optional


class APIResource(OpenAIObject):
    api_prefix = ""
    azure_api_prefix = "openai"
    azure_deployments_prefix = "deployments"

    @classmethod
    def retrieve(
        cls, id, api_key=None, request_id=None, request_timeout=None, **params
    ):
        instance = cls(id=id, api_key=api_key, **params)
        instance.refresh(request_id=request_id, request_timeout=request_timeout)
        return instance

    @classmethod
    def aretrieve(
        cls, id, api_key=None, request_id=None, request_timeout=None, **params
    ):
        instance = cls(id=id, api_key=api_key, **params)
        return instance.arefresh(request_id=request_id, request_timeout=request_timeout)

    def refresh(self, request_id=None, request_timeout=None):
        self.refresh_from(
            self.request(
                "get",
                self.instance_url(),
                request_id=request_id,
                request_timeout=request_timeout,
            )
        )
        return self

    async def arefresh(self, request_id=None, request_timeout=None):
        self.refresh_from(
            await self.arequest(
                "get",
                self.instance_url(operation="refresh"),
                request_id=request_id,
                request_timeout=request_timeout,
            )
        )
        return self

    @classmethod
    def class_url(cls):
        if cls == APIResource:
            raise NotImplementedError(
                "APIResource is an abstract class. You should perform actions on its subclasses."
            )
        # Namespaces are separated in object names with periods (.) and in URLs
        # with forward slashes (/), so replace the former with the latter.
        base = cls.OBJECT_NAME.replace(".", "/")  # type: ignore
        if cls.api_prefix:
            return "/%s/%s" % (cls.api_prefix, base)
        return "/%s" % (base)

    def instance_url(self, operation=None):
        id = self.get("id")

        if not isinstance(id, str):
            raise error.InvalidRequestError(
                "Could not determine which URL to request: %s instance "
                "has invalid ID: %r, %s. ID should be of type `str` (or"
                " `unicode`)" % (type(self).__name__, id, type(id)),
                "id",
            )
        api_version = self.api_version or openai.api_version
        extn = quote_plus(id)

        if self.typed_api_type in (ApiType.AZURE, ApiType.AZURE_AD):
            if not api_version:
                raise error.InvalidRequestError(
                    "An API version is required for the Azure API type."
                )

            if not operation:
                base = self.class_url()
                return "/%s%s/%s?api-version=%s" % (
                    self.azure_api_prefix,
                    base,
                    extn,
                    api_version,
                )

            return "/%s/%s/%s/%s?api-version=%s" % (
                self.azure_api_prefix,
                self.azure_deployments_prefix,
                extn,
                operation,
                api_version,
            )

        elif self.typed_api_type == ApiType.OPEN_AI:
            base = self.class_url()
            return "%s/%s" % (base, extn)

        else:
            raise error.InvalidAPIType("Unsupported API type %s" % self.api_type)

    # The `method_` and `url_` arguments are suffixed with an underscore to
    # avoid conflicting with actual request parameters in `params`.
    @classmethod
    def _static_request(
        cls,
        method_,
        url_,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        requestor = api_requestor.APIRequestor(
            api_key,
            api_version=api_version,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
        )
        response, _, api_key = requestor.request(
            method_, url_, params, request_id=request_id
        )
        return util.convert_to_openai_object(
            response, api_key, api_version, organization
        )

    @classmethod
    async def _astatic_request(
        cls,
        method_,
        url_,
        api_key=None,
        api_base=None,
        api_type=None,
        request_id=None,
        api_version=None,
        organization=None,
        **params,
    ):
        requestor = api_requestor.APIRequestor(
            api_key,
            api_version=api_version,
            organization=organization,
            api_base=api_base,
            api_type=api_type,
        )
        response, _, api_key = await requestor.arequest(
            method_, url_, params, request_id=request_id
        )
        return response

    @classmethod
    def _get_api_type_and_version(
        cls, api_type: Optional[str] = None, api_version: Optional[str] = None
    ):
        typed_api_type = (
            ApiType.from_str(api_type)
            if api_type
            else ApiType.from_str(openai.api_type)
        )
        typed_api_version = api_version or openai.api_version
        return (typed_api_type, typed_api_version)
