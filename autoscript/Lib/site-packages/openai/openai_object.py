import json
from copy import deepcopy
from typing import Optional, Tuple, Union

import openai
from openai import api_requestor, util
from openai.openai_response import OpenAIResponse
from openai.util import ApiType


class OpenAIObject(dict):
    api_base_override = None

    def __init__(
        self,
        id=None,
        api_key=None,
        api_version=None,
        api_type=None,
        organization=None,
        response_ms: Optional[int] = None,
        api_base=None,
        engine=None,
        **params,
    ):
        super(OpenAIObject, self).__init__()

        if response_ms is not None and not isinstance(response_ms, int):
            raise TypeError(f"response_ms is a {type(response_ms).__name__}.")
        self._response_ms = response_ms

        self._retrieve_params = params

        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "api_version", api_version)
        object.__setattr__(self, "api_type", api_type)
        object.__setattr__(self, "organization", organization)
        object.__setattr__(self, "api_base_override", api_base)
        object.__setattr__(self, "engine", engine)

        if id:
            self["id"] = id

    @property
    def response_ms(self) -> Optional[int]:
        return self._response_ms

    def __setattr__(self, k, v):
        if k[0] == "_" or k in self.__dict__:
            return super(OpenAIObject, self).__setattr__(k, v)

        self[k] = v
        return None

    def __getattr__(self, k):
        if k[0] == "_":
            raise AttributeError(k)
        try:
            return self[k]
        except KeyError as err:
            raise AttributeError(*err.args)

    def __delattr__(self, k):
        if k[0] == "_" or k in self.__dict__:
            return super(OpenAIObject, self).__delattr__(k)
        else:
            del self[k]

    def __setitem__(self, k, v):
        if v == "":
            raise ValueError(
                "You cannot set %s to an empty string. "
                "We interpret empty strings as None in requests."
                "You may set %s.%s = None to delete the property" % (k, str(self), k)
            )
        super(OpenAIObject, self).__setitem__(k, v)

    def __delitem__(self, k):
        raise NotImplementedError("del is not supported")

    # Custom unpickling method that uses `update` to update the dictionary
    # without calling __setitem__, which would fail if any value is an empty
    # string
    def __setstate__(self, state):
        self.update(state)

    # Custom pickling method to ensure the instance is pickled as a custom
    # class and not as a dict, otherwise __setstate__ would not be called when
    # unpickling.
    def __reduce__(self):
        reduce_value = (
            type(self),  # callable
            (  # args
                self.get("id", None),
                self.api_key,
                self.api_version,
                self.api_type,
                self.organization,
            ),
            dict(self),  # state
        )
        return reduce_value

    @classmethod
    def construct_from(
        cls,
        values,
        api_key: Optional[str] = None,
        api_version=None,
        organization=None,
        engine=None,
        response_ms: Optional[int] = None,
    ):
        instance = cls(
            values.get("id"),
            api_key=api_key,
            api_version=api_version,
            organization=organization,
            engine=engine,
            response_ms=response_ms,
        )
        instance.refresh_from(
            values,
            api_key=api_key,
            api_version=api_version,
            organization=organization,
            response_ms=response_ms,
        )
        return instance

    def refresh_from(
        self,
        values,
        api_key=None,
        api_version=None,
        api_type=None,
        organization=None,
        response_ms: Optional[int] = None,
    ):
        self.api_key = api_key or getattr(values, "api_key", None)
        self.api_version = api_version or getattr(values, "api_version", None)
        self.api_type = api_type or getattr(values, "api_type", None)
        self.organization = organization or getattr(values, "organization", None)
        self._response_ms = response_ms or getattr(values, "_response_ms", None)

        # Wipe old state before setting new.
        self.clear()
        for k, v in values.items():
            super(OpenAIObject, self).__setitem__(
                k, util.convert_to_openai_object(v, api_key, api_version, organization)
            )

        self._previous = values

    @classmethod
    def api_base(cls):
        return None

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        stream=False,
        plain_old_data=False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        if params is None:
            params = self._retrieve_params
        requestor = api_requestor.APIRequestor(
            key=self.api_key,
            api_base=self.api_base_override or self.api_base(),
            api_type=self.api_type,
            api_version=self.api_version,
            organization=self.organization,
        )
        response, stream, api_key = requestor.request(
            method,
            url,
            params=params,
            stream=stream,
            headers=headers,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            assert not isinstance(response, OpenAIResponse)  # must be an iterator
            return (
                util.convert_to_openai_object(
                    line,
                    api_key,
                    self.api_version,
                    self.organization,
                    plain_old_data=plain_old_data,
                )
                for line in response
            )
        else:
            return util.convert_to_openai_object(
                response,
                api_key,
                self.api_version,
                self.organization,
                plain_old_data=plain_old_data,
            )

    async def arequest(
        self,
        method,
        url,
        params=None,
        headers=None,
        stream=False,
        plain_old_data=False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        if params is None:
            params = self._retrieve_params
        requestor = api_requestor.APIRequestor(
            key=self.api_key,
            api_base=self.api_base_override or self.api_base(),
            api_type=self.api_type,
            api_version=self.api_version,
            organization=self.organization,
        )
        response, stream, api_key = await requestor.arequest(
            method,
            url,
            params=params,
            stream=stream,
            headers=headers,
            request_id=request_id,
            request_timeout=request_timeout,
        )

        if stream:
            assert not isinstance(response, OpenAIResponse)  # must be an iterator
            return (
                util.convert_to_openai_object(
                    line,
                    api_key,
                    self.api_version,
                    self.organization,
                    plain_old_data=plain_old_data,
                )
                for line in response
            )
        else:
            return util.convert_to_openai_object(
                response,
                api_key,
                self.api_version,
                self.organization,
                plain_old_data=plain_old_data,
            )

    def __repr__(self):
        ident_parts = [type(self).__name__]

        obj = self.get("object")
        if isinstance(obj, str):
            ident_parts.append(obj)

        if isinstance(self.get("id"), str):
            ident_parts.append("id=%s" % (self.get("id"),))

        unicode_repr = "<%s at %s> JSON: %s" % (
            " ".join(ident_parts),
            hex(id(self)),
            str(self),
        )

        return unicode_repr

    def __str__(self):
        obj = self.to_dict_recursive()
        return json.dumps(obj, sort_keys=True, indent=2)

    def to_dict(self):
        return dict(self)

    def to_dict_recursive(self):
        d = dict(self)
        for k, v in d.items():
            if isinstance(v, OpenAIObject):
                d[k] = v.to_dict_recursive()
            elif isinstance(v, list):
                d[k] = [
                    e.to_dict_recursive() if isinstance(e, OpenAIObject) else e
                    for e in v
                ]
        return d

    @property
    def openai_id(self):
        return self.id

    @property
    def typed_api_type(self):
        return (
            ApiType.from_str(self.api_type)
            if self.api_type
            else ApiType.from_str(openai.api_type)
        )

    # This class overrides __setitem__ to throw exceptions on inputs that it
    # doesn't like. This can cause problems when we try to copy an object
    # wholesale because some data that's returned from the API may not be valid
    # if it was set to be set manually. Here we override the class' copy
    # arguments so that we can bypass these possible exceptions on __setitem__.
    def __copy__(self):
        copied = OpenAIObject(
            self.get("id"),
            self.api_key,
            api_version=self.api_version,
            api_type=self.api_type,
            organization=self.organization,
        )

        copied._retrieve_params = self._retrieve_params

        for k, v in self.items():
            # Call parent's __setitem__ to avoid checks that we've added in the
            # overridden version that can throw exceptions.
            super(OpenAIObject, copied).__setitem__(k, v)

        return copied

    # This class overrides __setitem__ to throw exceptions on inputs that it
    # doesn't like. This can cause problems when we try to copy an object
    # wholesale because some data that's returned from the API may not be valid
    # if it was set to be set manually. Here we override the class' copy
    # arguments so that we can bypass these possible exceptions on __setitem__.
    def __deepcopy__(self, memo):
        copied = self.__copy__()
        memo[id(self)] = copied

        for k, v in self.items():
            # Call parent's __setitem__ to avoid checks that we've added in the
            # overridden version that can throw exceptions.
            super(OpenAIObject, copied).__setitem__(k, deepcopy(v, memo))

        return copied
