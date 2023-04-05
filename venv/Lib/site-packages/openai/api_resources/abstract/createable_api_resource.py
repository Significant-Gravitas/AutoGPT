from openai import api_requestor, util, error
from openai.api_resources.abstract.api_resource import APIResource
from openai.util import ApiType


class CreateableAPIResource(APIResource):
    plain_old_data = False

    @classmethod
    def __prepare_create_requestor(
        cls,
        api_key=None,
        api_base=None,
        api_type=None,
        api_version=None,
        organization=None,
    ):
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
            base = cls.class_url()
            url = "/%s%s?api-version=%s" % (cls.azure_api_prefix, base, api_version)
        elif typed_api_type == ApiType.OPEN_AI:
            url = cls.class_url()
        else:
            raise error.InvalidAPIType("Unsupported API type %s" % api_type)
        return requestor, url

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
        requestor, url = cls.__prepare_create_requestor(
            api_key,
            api_base,
            api_type,
            api_version,
            organization,
        )

        response, _, api_key = requestor.request(
            "post", url, params, request_id=request_id
        )

        return util.convert_to_openai_object(
            response,
            api_key,
            api_version,
            organization,
            plain_old_data=cls.plain_old_data,
        )

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
        requestor, url = cls.__prepare_create_requestor(
            api_key,
            api_base,
            api_type,
            api_version,
            organization,
        )

        response, _, api_key = await requestor.arequest(
            "post", url, params, request_id=request_id
        )

        return util.convert_to_openai_object(
            response,
            api_key,
            api_version,
            organization,
            plain_old_data=cls.plain_old_data,
        )
