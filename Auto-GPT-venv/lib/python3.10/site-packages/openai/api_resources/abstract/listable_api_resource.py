from openai import api_requestor, util, error
from openai.api_resources.abstract.api_resource import APIResource
from openai.util import ApiType


class ListableAPIResource(APIResource):
    @classmethod
    def auto_paging_iter(cls, *args, **params):
        return cls.list(*args, **params).auto_paging_iter()

    @classmethod
    def __prepare_list_requestor(
        cls,
        api_key=None,
        api_version=None,
        organization=None,
        api_base=None,
        api_type=None,
    ):
        requestor = api_requestor.APIRequestor(
            api_key,
            api_base=api_base or cls.api_base(),
            api_version=api_version,
            api_type=api_type,
            organization=organization,
        )

        typed_api_type, api_version = cls._get_api_type_and_version(
            api_type, api_version
        )

        if typed_api_type in (ApiType.AZURE, ApiType.AZURE_AD):
            base = cls.class_url()
            url = "/%s%s?api-version=%s" % (cls.azure_api_prefix, base, api_version)
        elif typed_api_type == ApiType.OPEN_AI:
            url = cls.class_url()
        else:
            raise error.InvalidAPIType("Unsupported API type %s" % api_type)
        return requestor, url

    @classmethod
    def list(
        cls,
        api_key=None,
        request_id=None,
        api_version=None,
        organization=None,
        api_base=None,
        api_type=None,
        **params,
    ):
        requestor, url = cls.__prepare_list_requestor(
            api_key,
            api_version,
            organization,
            api_base,
            api_type,
        )

        response, _, api_key = requestor.request(
            "get", url, params, request_id=request_id
        )
        openai_object = util.convert_to_openai_object(
            response, api_key, api_version, organization
        )
        openai_object._retrieve_params = params
        return openai_object

    @classmethod
    async def alist(
        cls,
        api_key=None,
        request_id=None,
        api_version=None,
        organization=None,
        api_base=None,
        api_type=None,
        **params,
    ):
        requestor, url = cls.__prepare_list_requestor(
            api_key,
            api_version,
            organization,
            api_base,
            api_type,
        )

        response, _, api_key = await requestor.arequest(
            "get", url, params, request_id=request_id
        )
        openai_object = util.convert_to_openai_object(
            response, api_key, api_version, organization
        )
        openai_object._retrieve_params = params
        return openai_object
