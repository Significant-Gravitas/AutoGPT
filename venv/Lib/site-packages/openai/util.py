import logging
import os
import re
import sys
from enum import Enum
from typing import Optional

import openai

OPENAI_LOG = os.environ.get("OPENAI_LOG")

logger = logging.getLogger("openai")

__all__ = [
    "log_info",
    "log_debug",
    "log_warn",
    "logfmt",
]

api_key_to_header = (
    lambda api, key: {"Authorization": f"Bearer {key}"}
    if api in (ApiType.OPEN_AI, ApiType.AZURE_AD)
    else {"api-key": f"{key}"}
)


class ApiType(Enum):
    AZURE = 1
    OPEN_AI = 2
    AZURE_AD = 3

    @staticmethod
    def from_str(label):
        if label.lower() == "azure":
            return ApiType.AZURE
        elif label.lower() in ("azure_ad", "azuread"):
            return ApiType.AZURE_AD
        elif label.lower() in ("open_ai", "openai"):
            return ApiType.OPEN_AI
        else:
            raise openai.error.InvalidAPIType(
                "The API type provided in invalid. Please select one of the supported API types: 'azure', 'azure_ad', 'open_ai'"
            )


def _console_log_level():
    if openai.log in ["debug", "info"]:
        return openai.log
    elif OPENAI_LOG in ["debug", "info"]:
        return OPENAI_LOG
    else:
        return None


def log_debug(message, **params):
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() == "debug":
        print(msg, file=sys.stderr)
    logger.debug(msg)


def log_info(message, **params):
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() in ["debug", "info"]:
        print(msg, file=sys.stderr)
    logger.info(msg)


def log_warn(message, **params):
    msg = logfmt(dict(message=message, **params))
    print(msg, file=sys.stderr)
    logger.warn(msg)


def logfmt(props):
    def fmt(key, val):
        # Handle case where val is a bytes or bytesarray
        if hasattr(val, "decode"):
            val = val.decode("utf-8")
        # Check if val is already a string to avoid re-encoding into ascii.
        if not isinstance(val, str):
            val = str(val)
        if re.search(r"\s", val):
            val = repr(val)
        # key should already be a string
        if re.search(r"\s", key):
            key = repr(key)
        return "{key}={val}".format(key=key, val=val)

    return " ".join([fmt(key, val) for key, val in sorted(props.items())])


def get_object_classes():
    # This is here to avoid a circular dependency
    from openai.object_classes import OBJECT_CLASSES

    return OBJECT_CLASSES


def convert_to_openai_object(
    resp,
    api_key=None,
    api_version=None,
    organization=None,
    engine=None,
    plain_old_data=False,
):
    # If we get a OpenAIResponse, we'll want to return a OpenAIObject.

    response_ms: Optional[int] = None
    if isinstance(resp, openai.openai_response.OpenAIResponse):
        organization = resp.organization
        response_ms = resp.response_ms
        resp = resp.data

    if plain_old_data:
        return resp
    elif isinstance(resp, list):
        return [
            convert_to_openai_object(
                i, api_key, api_version, organization, engine=engine
            )
            for i in resp
        ]
    elif isinstance(resp, dict) and not isinstance(
        resp, openai.openai_object.OpenAIObject
    ):
        resp = resp.copy()
        klass_name = resp.get("object")
        if isinstance(klass_name, str):
            klass = get_object_classes().get(
                klass_name, openai.openai_object.OpenAIObject
            )
        else:
            klass = openai.openai_object.OpenAIObject

        return klass.construct_from(
            resp,
            api_key=api_key,
            api_version=api_version,
            organization=organization,
            response_ms=response_ms,
            engine=engine,
        )
    else:
        return resp


def convert_to_dict(obj):
    """Converts a OpenAIObject back to a regular dict.

    Nested OpenAIObjects are also converted back to regular dicts.

    :param obj: The OpenAIObject to convert.

    :returns: The OpenAIObject as a dict.
    """
    if isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    # This works by virtue of the fact that OpenAIObjects _are_ dicts. The dict
    # comprehension returns a regular dict and recursively applies the
    # conversion to each value.
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def default_api_key() -> str:
    if openai.api_key_path:
        with open(openai.api_key_path, "rt") as k:
            api_key = k.read().strip()
            if not api_key.startswith("sk-"):
                raise ValueError(f"Malformed API key in {openai.api_key_path}.")
            return api_key
    elif openai.api_key is not None:
        return openai.api_key
    else:
        raise openai.error.AuthenticationError(
            "No API key provided. You can set your API key in code using 'openai.api_key = <API-KEY>', or you can set the environment variable OPENAI_API_KEY=<API-KEY>). If your API key is stored in a file, you can point the openai module at it with 'openai.api_key_path = <PATH>'. You can generate API keys in the OpenAI web interface. See https://onboard.openai.com for details, or email support@openai.com if you have any questions."
        )
