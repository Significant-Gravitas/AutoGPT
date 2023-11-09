# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client for discovery based APIs.

A client library for Google's discovery based APIs.
"""
from __future__ import absolute_import

__author__ = "jcgregorio@google.com (Joe Gregorio)"
__all__ = ["build", "build_from_document", "fix_method_name", "key2param"]

from collections import OrderedDict
import collections.abc

# Standard library imports
import copy
from email.generator import BytesGenerator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
import http.client as http_client
import io
import json
import keyword
import logging
import mimetypes
import os
import re
import urllib

import google.api_core.client_options
from google.auth.exceptions import MutualTLSChannelError
from google.auth.transport import mtls
from google.oauth2 import service_account

# Third-party imports
import httplib2
import uritemplate

try:
    import google_auth_httplib2
except ImportError:  # pragma: NO COVER
    google_auth_httplib2 = None

# Local imports
from googleapiclient import _auth, mimeparse
from googleapiclient._helpers import _add_query_parameter, positional
from googleapiclient.errors import (
    HttpError,
    InvalidJsonError,
    MediaUploadSizeError,
    UnacceptableMimeTypeError,
    UnknownApiNameOrVersion,
    UnknownFileType,
)
from googleapiclient.http import (
    BatchHttpRequest,
    HttpMock,
    HttpMockSequence,
    HttpRequest,
    MediaFileUpload,
    MediaUpload,
    build_http,
)
from googleapiclient.model import JsonModel, MediaModel, RawModel
from googleapiclient.schema import Schemas

# The client library requires a version of httplib2 that supports RETRIES.
httplib2.RETRIES = 1

logger = logging.getLogger(__name__)

URITEMPLATE = re.compile("{[^}]*}")
VARNAME = re.compile("[a-zA-Z0-9_-]+")
DISCOVERY_URI = (
    "https://www.googleapis.com/discovery/v1/apis/" "{api}/{apiVersion}/rest"
)
V1_DISCOVERY_URI = DISCOVERY_URI
V2_DISCOVERY_URI = (
    "https://{api}.googleapis.com/$discovery/rest?" "version={apiVersion}"
)
DEFAULT_METHOD_DOC = "A description of how to use this function"
HTTP_PAYLOAD_METHODS = frozenset(["PUT", "POST", "PATCH"])

_MEDIA_SIZE_BIT_SHIFTS = {"KB": 10, "MB": 20, "GB": 30, "TB": 40}
BODY_PARAMETER_DEFAULT_VALUE = {"description": "The request body.", "type": "object"}
MEDIA_BODY_PARAMETER_DEFAULT_VALUE = {
    "description": (
        "The filename of the media request body, or an instance "
        "of a MediaUpload object."
    ),
    "type": "string",
    "required": False,
}
MEDIA_MIME_TYPE_PARAMETER_DEFAULT_VALUE = {
    "description": (
        "The MIME type of the media request body, or an instance "
        "of a MediaUpload object."
    ),
    "type": "string",
    "required": False,
}
_PAGE_TOKEN_NAMES = ("pageToken", "nextPageToken")

# Parameters controlling mTLS behavior. See https://google.aip.dev/auth/4114.
GOOGLE_API_USE_CLIENT_CERTIFICATE = "GOOGLE_API_USE_CLIENT_CERTIFICATE"
GOOGLE_API_USE_MTLS_ENDPOINT = "GOOGLE_API_USE_MTLS_ENDPOINT"

# Parameters accepted by the stack, but not visible via discovery.
# TODO(dhermes): Remove 'userip' in 'v2'.
STACK_QUERY_PARAMETERS = frozenset(["trace", "pp", "userip", "strict"])
STACK_QUERY_PARAMETER_DEFAULT_VALUE = {"type": "string", "location": "query"}

# Library-specific reserved words beyond Python keywords.
RESERVED_WORDS = frozenset(["body"])

# patch _write_lines to avoid munging '\r' into '\n'
# ( https://bugs.python.org/issue18886 https://bugs.python.org/issue19003 )
class _BytesGenerator(BytesGenerator):
    _write_lines = BytesGenerator.write


def fix_method_name(name):
    """Fix method names to avoid '$' characters and reserved word conflicts.

    Args:
      name: string, method name.

    Returns:
      The name with '_' appended if the name is a reserved word and '$' and '-'
      replaced with '_'.
    """
    name = name.replace("$", "_").replace("-", "_")
    if keyword.iskeyword(name) or name in RESERVED_WORDS:
        return name + "_"
    else:
        return name


def key2param(key):
    """Converts key names into parameter names.

    For example, converting "max-results" -> "max_results"

    Args:
      key: string, the method key name.

    Returns:
      A safe method name based on the key name.
    """
    result = []
    key = list(key)
    if not key[0].isalpha():
        result.append("x")
    for c in key:
        if c.isalnum():
            result.append(c)
        else:
            result.append("_")

    return "".join(result)


@positional(2)
def build(
    serviceName,
    version,
    http=None,
    discoveryServiceUrl=None,
    developerKey=None,
    model=None,
    requestBuilder=HttpRequest,
    credentials=None,
    cache_discovery=True,
    cache=None,
    client_options=None,
    adc_cert_path=None,
    adc_key_path=None,
    num_retries=1,
    static_discovery=None,
    always_use_jwt_access=False,
):
    """Construct a Resource for interacting with an API.

    Construct a Resource object for interacting with an API. The serviceName and
    version are the names from the Discovery service.

    Args:
      serviceName: string, name of the service.
      version: string, the version of the service.
      http: httplib2.Http, An instance of httplib2.Http or something that acts
        like it that HTTP requests will be made through.
      discoveryServiceUrl: string, a URI Template that points to the location of
        the discovery service. It should have two parameters {api} and
        {apiVersion} that when filled in produce an absolute URI to the discovery
        document for that service.
      developerKey: string, key obtained from
        https://code.google.com/apis/console.
      model: googleapiclient.Model, converts to and from the wire format.
      requestBuilder: googleapiclient.http.HttpRequest, encapsulator for an HTTP
        request.
      credentials: oauth2client.Credentials or
        google.auth.credentials.Credentials, credentials to be used for
        authentication.
      cache_discovery: Boolean, whether or not to cache the discovery doc.
      cache: googleapiclient.discovery_cache.base.CacheBase, an optional
        cache object for the discovery documents.
      client_options: Mapping object or google.api_core.client_options, client
        options to set user options on the client.
        (1) The API endpoint should be set through client_options. If API endpoint
        is not set, `GOOGLE_API_USE_MTLS_ENDPOINT` environment variable can be used
        to control which endpoint to use.
        (2) client_cert_source is not supported, client cert should be provided using
        client_encrypted_cert_source instead. In order to use the provided client
        cert, `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be
        set to `true`.
        More details on the environment variables are here:
        https://google.aip.dev/auth/4114
      adc_cert_path: str, client certificate file path to save the application
        default client certificate for mTLS. This field is required if you want to
        use the default client certificate. `GOOGLE_API_USE_CLIENT_CERTIFICATE`
        environment variable must be set to `true` in order to use this field,
        otherwise this field doesn't nothing.
        More details on the environment variables are here:
        https://google.aip.dev/auth/4114
      adc_key_path: str, client encrypted private key file path to save the
        application default client encrypted private key for mTLS. This field is
        required if you want to use the default client certificate.
        `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be set to
        `true` in order to use this field, otherwise this field doesn't nothing.
        More details on the environment variables are here:
        https://google.aip.dev/auth/4114
      num_retries: Integer, number of times to retry discovery with
        randomized exponential backoff in case of intermittent/connection issues.
      static_discovery: Boolean, whether or not to use the static discovery docs
        included in the library. The default value for `static_discovery` depends
        on the value of `discoveryServiceUrl`. `static_discovery` will default to
        `True` when `discoveryServiceUrl` is also not provided, otherwise it will
        default to `False`.
      always_use_jwt_access: Boolean, whether always use self signed JWT for service
        account credentials. This only applies to
        google.oauth2.service_account.Credentials.

    Returns:
      A Resource object with methods for interacting with the service.

    Raises:
      google.auth.exceptions.MutualTLSChannelError: if there are any problems
        setting up mutual TLS channel.
    """
    params = {"api": serviceName, "apiVersion": version}

    # The default value for `static_discovery` depends on the value of
    # `discoveryServiceUrl`. `static_discovery` will default to `True` when
    # `discoveryServiceUrl` is also not provided, otherwise it will default to
    # `False`. This is added for backwards compatability with
    # google-api-python-client 1.x which does not support the `static_discovery`
    # parameter.
    if static_discovery is None:
        if discoveryServiceUrl is None:
            static_discovery = True
        else:
            static_discovery = False

    if http is None:
        discovery_http = build_http()
    else:
        discovery_http = http

    service = None

    for discovery_url in _discovery_service_uri_options(discoveryServiceUrl, version):
        requested_url = uritemplate.expand(discovery_url, params)

        try:
            content = _retrieve_discovery_doc(
                requested_url,
                discovery_http,
                cache_discovery,
                serviceName,
                version,
                cache,
                developerKey,
                num_retries=num_retries,
                static_discovery=static_discovery,
            )
            service = build_from_document(
                content,
                base=discovery_url,
                http=http,
                developerKey=developerKey,
                model=model,
                requestBuilder=requestBuilder,
                credentials=credentials,
                client_options=client_options,
                adc_cert_path=adc_cert_path,
                adc_key_path=adc_key_path,
                always_use_jwt_access=always_use_jwt_access,
            )
            break  # exit if a service was created
        except HttpError as e:
            if e.resp.status == http_client.NOT_FOUND:
                continue
            else:
                raise e

    # If discovery_http was created by this function, we are done with it
    # and can safely close it
    if http is None:
        discovery_http.close()

    if service is None:
        raise UnknownApiNameOrVersion("name: %s  version: %s" % (serviceName, version))
    else:
        return service


def _discovery_service_uri_options(discoveryServiceUrl, version):
    """
      Returns Discovery URIs to be used for attempting to build the API Resource.

    Args:
      discoveryServiceUrl:
          string, the Original Discovery Service URL preferred by the customer.
      version:
          string, API Version requested

    Returns:
        A list of URIs to be tried for the Service Discovery, in order.
    """

    if discoveryServiceUrl is not None:
        return [discoveryServiceUrl]
    if version is None:
        # V1 Discovery won't work if the requested version is None
        logger.warning(
            "Discovery V1 does not support empty versions. Defaulting to V2..."
        )
        return [V2_DISCOVERY_URI]
    else:
        return [DISCOVERY_URI, V2_DISCOVERY_URI]


def _retrieve_discovery_doc(
    url,
    http,
    cache_discovery,
    serviceName,
    version,
    cache=None,
    developerKey=None,
    num_retries=1,
    static_discovery=True,
):
    """Retrieves the discovery_doc from cache or the internet.

    Args:
      url: string, the URL of the discovery document.
      http: httplib2.Http, An instance of httplib2.Http or something that acts
        like it through which HTTP requests will be made.
      cache_discovery: Boolean, whether or not to cache the discovery doc.
      serviceName: string, name of the service.
      version: string, the version of the service.
      cache: googleapiclient.discovery_cache.base.Cache, an optional cache
        object for the discovery documents.
      developerKey: string, Key for controlling API usage, generated
        from the API Console.
      num_retries: Integer, number of times to retry discovery with
        randomized exponential backoff in case of intermittent/connection issues.
      static_discovery: Boolean, whether or not to use the static discovery docs
        included in the library.

    Returns:
      A unicode string representation of the discovery document.
    """
    from . import discovery_cache

    if cache_discovery:
        if cache is None:
            cache = discovery_cache.autodetect()
        if cache:
            content = cache.get(url)
            if content:
                return content

    # When `static_discovery=True`, use static discovery artifacts included
    # with the library
    if static_discovery:
        content = discovery_cache.get_static_doc(serviceName, version)
        if content:
            return content
        else:
            raise UnknownApiNameOrVersion(
                "name: %s  version: %s" % (serviceName, version)
            )

    actual_url = url
    # REMOTE_ADDR is defined by the CGI spec [RFC3875] as the environment
    # variable that contains the network address of the client sending the
    # request. If it exists then add that to the request for the discovery
    # document to avoid exceeding the quota on discovery requests.
    if "REMOTE_ADDR" in os.environ:
        actual_url = _add_query_parameter(url, "userIp", os.environ["REMOTE_ADDR"])
    if developerKey:
        actual_url = _add_query_parameter(url, "key", developerKey)
    logger.debug("URL being requested: GET %s", actual_url)

    # Execute this request with retries build into HttpRequest
    # Note that it will already raise an error if we don't get a 2xx response
    req = HttpRequest(http, HttpRequest.null_postproc, actual_url)
    resp, content = req.execute(num_retries=num_retries)

    try:
        content = content.decode("utf-8")
    except AttributeError:
        pass

    try:
        service = json.loads(content)
    except ValueError as e:
        logger.error("Failed to parse as JSON: " + content)
        raise InvalidJsonError()
    if cache_discovery and cache:
        cache.set(url, content)
    return content


@positional(1)
def build_from_document(
    service,
    base=None,
    future=None,
    http=None,
    developerKey=None,
    model=None,
    requestBuilder=HttpRequest,
    credentials=None,
    client_options=None,
    adc_cert_path=None,
    adc_key_path=None,
    always_use_jwt_access=False,
):
    """Create a Resource for interacting with an API.

    Same as `build()`, but constructs the Resource object from a discovery
    document that is it given, as opposed to retrieving one over HTTP.

    Args:
      service: string or object, the JSON discovery document describing the API.
        The value passed in may either be the JSON string or the deserialized
        JSON.
      base: string, base URI for all HTTP requests, usually the discovery URI.
        This parameter is no longer used as rootUrl and servicePath are included
        within the discovery document. (deprecated)
      future: string, discovery document with future capabilities (deprecated).
      http: httplib2.Http, An instance of httplib2.Http or something that acts
        like it that HTTP requests will be made through.
      developerKey: string, Key for controlling API usage, generated
        from the API Console.
      model: Model class instance that serializes and de-serializes requests and
        responses.
      requestBuilder: Takes an http request and packages it up to be executed.
      credentials: oauth2client.Credentials or
        google.auth.credentials.Credentials, credentials to be used for
        authentication.
      client_options: Mapping object or google.api_core.client_options, client
        options to set user options on the client.
        (1) The API endpoint should be set through client_options. If API endpoint
        is not set, `GOOGLE_API_USE_MTLS_ENDPOINT` environment variable can be used
        to control which endpoint to use.
        (2) client_cert_source is not supported, client cert should be provided using
        client_encrypted_cert_source instead. In order to use the provided client
        cert, `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be
        set to `true`.
        More details on the environment variables are here:
        https://google.aip.dev/auth/4114
      adc_cert_path: str, client certificate file path to save the application
        default client certificate for mTLS. This field is required if you want to
        use the default client certificate. `GOOGLE_API_USE_CLIENT_CERTIFICATE`
        environment variable must be set to `true` in order to use this field,
        otherwise this field doesn't nothing.
        More details on the environment variables are here:
        https://google.aip.dev/auth/4114
      adc_key_path: str, client encrypted private key file path to save the
        application default client encrypted private key for mTLS. This field is
        required if you want to use the default client certificate.
        `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable must be set to
        `true` in order to use this field, otherwise this field doesn't nothing.
        More details on the environment variables are here:
        https://google.aip.dev/auth/4114
      always_use_jwt_access: Boolean, whether always use self signed JWT for service
        account credentials. This only applies to
        google.oauth2.service_account.Credentials.

    Returns:
      A Resource object with methods for interacting with the service.

    Raises:
      google.auth.exceptions.MutualTLSChannelError: if there are any problems
        setting up mutual TLS channel.
    """

    if client_options is None:
        client_options = google.api_core.client_options.ClientOptions()
    if isinstance(client_options, collections.abc.Mapping):
        client_options = google.api_core.client_options.from_dict(client_options)

    if http is not None:
        # if http is passed, the user cannot provide credentials
        banned_options = [
            (credentials, "credentials"),
            (client_options.credentials_file, "client_options.credentials_file"),
        ]
        for option, name in banned_options:
            if option is not None:
                raise ValueError(
                    "Arguments http and {} are mutually exclusive".format(name)
                )

    if isinstance(service, str):
        service = json.loads(service)
    elif isinstance(service, bytes):
        service = json.loads(service.decode("utf-8"))

    if "rootUrl" not in service and isinstance(http, (HttpMock, HttpMockSequence)):
        logger.error(
            "You are using HttpMock or HttpMockSequence without"
            + "having the service discovery doc in cache. Try calling "
            + "build() without mocking once first to populate the "
            + "cache."
        )
        raise InvalidJsonError()

    # If an API Endpoint is provided on client options, use that as the base URL
    base = urllib.parse.urljoin(service["rootUrl"], service["servicePath"])
    audience_for_self_signed_jwt = base
    if client_options.api_endpoint:
        base = client_options.api_endpoint

    schema = Schemas(service)

    # If the http client is not specified, then we must construct an http client
    # to make requests. If the service has scopes, then we also need to setup
    # authentication.
    if http is None:
        # Does the service require scopes?
        scopes = list(
            service.get("auth", {}).get("oauth2", {}).get("scopes", {}).keys()
        )

        # If so, then the we need to setup authentication if no developerKey is
        # specified.
        if scopes and not developerKey:
            # Make sure the user didn't pass multiple credentials
            if client_options.credentials_file and credentials:
                raise google.api_core.exceptions.DuplicateCredentialArgs(
                    "client_options.credentials_file and credentials are mutually exclusive."
                )
            # Check for credentials file via client options
            if client_options.credentials_file:
                credentials = _auth.credentials_from_file(
                    client_options.credentials_file,
                    scopes=client_options.scopes,
                    quota_project_id=client_options.quota_project_id,
                )
            # If the user didn't pass in credentials, attempt to acquire application
            # default credentials.
            if credentials is None:
                credentials = _auth.default_credentials(
                    scopes=client_options.scopes,
                    quota_project_id=client_options.quota_project_id,
                )

            # The credentials need to be scoped.
            # If the user provided scopes via client_options don't override them
            if not client_options.scopes:
                credentials = _auth.with_scopes(credentials, scopes)

        # For google-auth service account credentials, enable self signed JWT if
        # always_use_jwt_access is true.
        if (
            credentials
            and isinstance(credentials, service_account.Credentials)
            and always_use_jwt_access
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(always_use_jwt_access)
            credentials._create_self_signed_jwt(audience_for_self_signed_jwt)

        # If credentials are provided, create an authorized http instance;
        # otherwise, skip authentication.
        if credentials:
            http = _auth.authorized_http(credentials)

        # If the service doesn't require scopes then there is no need for
        # authentication.
        else:
            http = build_http()

        # Obtain client cert and create mTLS http channel if cert exists.
        client_cert_to_use = None
        use_client_cert = os.getenv(GOOGLE_API_USE_CLIENT_CERTIFICATE, "false")
        if not use_client_cert in ("true", "false"):
            raise MutualTLSChannelError(
                "Unsupported GOOGLE_API_USE_CLIENT_CERTIFICATE value. Accepted values: true, false"
            )
        if client_options and client_options.client_cert_source:
            raise MutualTLSChannelError(
                "ClientOptions.client_cert_source is not supported, please use ClientOptions.client_encrypted_cert_source."
            )
        if use_client_cert == "true":
            if (
                client_options
                and hasattr(client_options, "client_encrypted_cert_source")
                and client_options.client_encrypted_cert_source
            ):
                client_cert_to_use = client_options.client_encrypted_cert_source
            elif (
                adc_cert_path and adc_key_path and mtls.has_default_client_cert_source()
            ):
                client_cert_to_use = mtls.default_client_encrypted_cert_source(
                    adc_cert_path, adc_key_path
                )
        if client_cert_to_use:
            cert_path, key_path, passphrase = client_cert_to_use()

            # The http object we built could be google_auth_httplib2.AuthorizedHttp
            # or httplib2.Http. In the first case we need to extract the wrapped
            # httplib2.Http object from google_auth_httplib2.AuthorizedHttp.
            http_channel = (
                http.http
                if google_auth_httplib2
                and isinstance(http, google_auth_httplib2.AuthorizedHttp)
                else http
            )
            http_channel.add_certificate(key_path, cert_path, "", passphrase)

        # If user doesn't provide api endpoint via client options, decide which
        # api endpoint to use.
        if "mtlsRootUrl" in service and (
            not client_options or not client_options.api_endpoint
        ):
            mtls_endpoint = urllib.parse.urljoin(
                service["mtlsRootUrl"], service["servicePath"]
            )
            use_mtls_endpoint = os.getenv(GOOGLE_API_USE_MTLS_ENDPOINT, "auto")

            if not use_mtls_endpoint in ("never", "auto", "always"):
                raise MutualTLSChannelError(
                    "Unsupported GOOGLE_API_USE_MTLS_ENDPOINT value. Accepted values: never, auto, always"
                )

            # Switch to mTLS endpoint, if environment variable is "always", or
            # environment varibable is "auto" and client cert exists.
            if use_mtls_endpoint == "always" or (
                use_mtls_endpoint == "auto" and client_cert_to_use
            ):
                base = mtls_endpoint

    if model is None:
        features = service.get("features", [])
        model = JsonModel("dataWrapper" in features)

    return Resource(
        http=http,
        baseUrl=base,
        model=model,
        developerKey=developerKey,
        requestBuilder=requestBuilder,
        resourceDesc=service,
        rootDesc=service,
        schema=schema,
    )


def _cast(value, schema_type):
    """Convert value to a string based on JSON Schema type.

    See http://tools.ietf.org/html/draft-zyp-json-schema-03 for more details on
    JSON Schema.

    Args:
      value: any, the value to convert
      schema_type: string, the type that value should be interpreted as

    Returns:
      A string representation of 'value' based on the schema_type.
    """
    if schema_type == "string":
        if type(value) == type("") or type(value) == type(""):
            return value
        else:
            return str(value)
    elif schema_type == "integer":
        return str(int(value))
    elif schema_type == "number":
        return str(float(value))
    elif schema_type == "boolean":
        return str(bool(value)).lower()
    else:
        if type(value) == type("") or type(value) == type(""):
            return value
        else:
            return str(value)


def _media_size_to_long(maxSize):
    """Convert a string media size, such as 10GB or 3TB into an integer.

    Args:
      maxSize: string, size as a string, such as 2MB or 7GB.

    Returns:
      The size as an integer value.
    """
    if len(maxSize) < 2:
        return 0
    units = maxSize[-2:].upper()
    bit_shift = _MEDIA_SIZE_BIT_SHIFTS.get(units)
    if bit_shift is not None:
        return int(maxSize[:-2]) << bit_shift
    else:
        return int(maxSize)


def _media_path_url_from_info(root_desc, path_url):
    """Creates an absolute media path URL.

    Constructed using the API root URI and service path from the discovery
    document and the relative path for the API method.

    Args:
      root_desc: Dictionary; the entire original deserialized discovery document.
      path_url: String; the relative URL for the API method. Relative to the API
          root, which is specified in the discovery document.

    Returns:
      String; the absolute URI for media upload for the API method.
    """
    return "%(root)supload/%(service_path)s%(path)s" % {
        "root": root_desc["rootUrl"],
        "service_path": root_desc["servicePath"],
        "path": path_url,
    }


def _fix_up_parameters(method_desc, root_desc, http_method, schema):
    """Updates parameters of an API method with values specific to this library.

    Specifically, adds whatever global parameters are specified by the API to the
    parameters for the individual method. Also adds parameters which don't
    appear in the discovery document, but are available to all discovery based
    APIs (these are listed in STACK_QUERY_PARAMETERS).

    SIDE EFFECTS: This updates the parameters dictionary object in the method
    description.

    Args:
      method_desc: Dictionary with metadata describing an API method. Value comes
          from the dictionary of methods stored in the 'methods' key in the
          deserialized discovery document.
      root_desc: Dictionary; the entire original deserialized discovery document.
      http_method: String; the HTTP method used to call the API method described
          in method_desc.
      schema: Object, mapping of schema names to schema descriptions.

    Returns:
      The updated Dictionary stored in the 'parameters' key of the method
          description dictionary.
    """
    parameters = method_desc.setdefault("parameters", {})

    # Add in the parameters common to all methods.
    for name, description in root_desc.get("parameters", {}).items():
        parameters[name] = description

    # Add in undocumented query parameters.
    for name in STACK_QUERY_PARAMETERS:
        parameters[name] = STACK_QUERY_PARAMETER_DEFAULT_VALUE.copy()

    # Add 'body' (our own reserved word) to parameters if the method supports
    # a request payload.
    if http_method in HTTP_PAYLOAD_METHODS and "request" in method_desc:
        body = BODY_PARAMETER_DEFAULT_VALUE.copy()
        body.update(method_desc["request"])
        parameters["body"] = body

    return parameters


def _fix_up_media_upload(method_desc, root_desc, path_url, parameters):
    """Adds 'media_body' and 'media_mime_type' parameters if supported by method.

    SIDE EFFECTS: If there is a 'mediaUpload' in the method description, adds
    'media_upload' key to parameters.

    Args:
      method_desc: Dictionary with metadata describing an API method. Value comes
          from the dictionary of methods stored in the 'methods' key in the
          deserialized discovery document.
      root_desc: Dictionary; the entire original deserialized discovery document.
      path_url: String; the relative URL for the API method. Relative to the API
          root, which is specified in the discovery document.
      parameters: A dictionary describing method parameters for method described
          in method_desc.

    Returns:
      Triple (accept, max_size, media_path_url) where:
        - accept is a list of strings representing what content types are
          accepted for media upload. Defaults to empty list if not in the
          discovery document.
        - max_size is a long representing the max size in bytes allowed for a
          media upload. Defaults to 0L if not in the discovery document.
        - media_path_url is a String; the absolute URI for media upload for the
          API method. Constructed using the API root URI and service path from
          the discovery document and the relative path for the API method. If
          media upload is not supported, this is None.
    """
    media_upload = method_desc.get("mediaUpload", {})
    accept = media_upload.get("accept", [])
    max_size = _media_size_to_long(media_upload.get("maxSize", ""))
    media_path_url = None

    if media_upload:
        media_path_url = _media_path_url_from_info(root_desc, path_url)
        parameters["media_body"] = MEDIA_BODY_PARAMETER_DEFAULT_VALUE.copy()
        parameters["media_mime_type"] = MEDIA_MIME_TYPE_PARAMETER_DEFAULT_VALUE.copy()

    return accept, max_size, media_path_url


def _fix_up_method_description(method_desc, root_desc, schema):
    """Updates a method description in a discovery document.

    SIDE EFFECTS: Changes the parameters dictionary in the method description with
    extra parameters which are used locally.

    Args:
      method_desc: Dictionary with metadata describing an API method. Value comes
          from the dictionary of methods stored in the 'methods' key in the
          deserialized discovery document.
      root_desc: Dictionary; the entire original deserialized discovery document.
      schema: Object, mapping of schema names to schema descriptions.

    Returns:
      Tuple (path_url, http_method, method_id, accept, max_size, media_path_url)
      where:
        - path_url is a String; the relative URL for the API method. Relative to
          the API root, which is specified in the discovery document.
        - http_method is a String; the HTTP method used to call the API method
          described in the method description.
        - method_id is a String; the name of the RPC method associated with the
          API method, and is in the method description in the 'id' key.
        - accept is a list of strings representing what content types are
          accepted for media upload. Defaults to empty list if not in the
          discovery document.
        - max_size is a long representing the max size in bytes allowed for a
          media upload. Defaults to 0L if not in the discovery document.
        - media_path_url is a String; the absolute URI for media upload for the
          API method. Constructed using the API root URI and service path from
          the discovery document and the relative path for the API method. If
          media upload is not supported, this is None.
    """
    path_url = method_desc["path"]
    http_method = method_desc["httpMethod"]
    method_id = method_desc["id"]

    parameters = _fix_up_parameters(method_desc, root_desc, http_method, schema)
    # Order is important. `_fix_up_media_upload` needs `method_desc` to have a
    # 'parameters' key and needs to know if there is a 'body' parameter because it
    # also sets a 'media_body' parameter.
    accept, max_size, media_path_url = _fix_up_media_upload(
        method_desc, root_desc, path_url, parameters
    )

    return path_url, http_method, method_id, accept, max_size, media_path_url


def _fix_up_media_path_base_url(media_path_url, base_url):
    """
    Update the media upload base url if its netloc doesn't match base url netloc.

    This can happen in case the base url was overridden by
    client_options.api_endpoint.

    Args:
      media_path_url: String; the absolute URI for media upload.
      base_url: string, base URL for the API. All requests are relative to this URI.

    Returns:
      String; the absolute URI for media upload.
    """
    parsed_media_url = urllib.parse.urlparse(media_path_url)
    parsed_base_url = urllib.parse.urlparse(base_url)
    if parsed_media_url.netloc == parsed_base_url.netloc:
        return media_path_url
    return urllib.parse.urlunparse(
        parsed_media_url._replace(netloc=parsed_base_url.netloc)
    )


def _urljoin(base, url):
    """Custom urljoin replacement supporting : before / in url."""
    # In general, it's unsafe to simply join base and url. However, for
    # the case of discovery documents, we know:
    #  * base will never contain params, query, or fragment
    #  * url will never contain a scheme or net_loc.
    # In general, this means we can safely join on /; we just need to
    # ensure we end up with precisely one / joining base and url. The
    # exception here is the case of media uploads, where url will be an
    # absolute url.
    if url.startswith("http://") or url.startswith("https://"):
        return urllib.parse.urljoin(base, url)
    new_base = base if base.endswith("/") else base + "/"
    new_url = url[1:] if url.startswith("/") else url
    return new_base + new_url


# TODO(dhermes): Convert this class to ResourceMethod and make it callable
class ResourceMethodParameters(object):
    """Represents the parameters associated with a method.

    Attributes:
      argmap: Map from method parameter name (string) to query parameter name
          (string).
      required_params: List of required parameters (represented by parameter
          name as string).
      repeated_params: List of repeated parameters (represented by parameter
          name as string).
      pattern_params: Map from method parameter name (string) to regular
          expression (as a string). If the pattern is set for a parameter, the
          value for that parameter must match the regular expression.
      query_params: List of parameters (represented by parameter name as string)
          that will be used in the query string.
      path_params: Set of parameters (represented by parameter name as string)
          that will be used in the base URL path.
      param_types: Map from method parameter name (string) to parameter type. Type
          can be any valid JSON schema type; valid values are 'any', 'array',
          'boolean', 'integer', 'number', 'object', or 'string'. Reference:
          http://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.1
      enum_params: Map from method parameter name (string) to list of strings,
         where each list of strings is the list of acceptable enum values.
    """

    def __init__(self, method_desc):
        """Constructor for ResourceMethodParameters.

        Sets default values and defers to set_parameters to populate.

        Args:
          method_desc: Dictionary with metadata describing an API method. Value
              comes from the dictionary of methods stored in the 'methods' key in
              the deserialized discovery document.
        """
        self.argmap = {}
        self.required_params = []
        self.repeated_params = []
        self.pattern_params = {}
        self.query_params = []
        # TODO(dhermes): Change path_params to a list if the extra URITEMPLATE
        #                parsing is gotten rid of.
        self.path_params = set()
        self.param_types = {}
        self.enum_params = {}

        self.set_parameters(method_desc)

    def set_parameters(self, method_desc):
        """Populates maps and lists based on method description.

        Iterates through each parameter for the method and parses the values from
        the parameter dictionary.

        Args:
          method_desc: Dictionary with metadata describing an API method. Value
              comes from the dictionary of methods stored in the 'methods' key in
              the deserialized discovery document.
        """
        parameters = method_desc.get("parameters", {})
        sorted_parameters = OrderedDict(sorted(parameters.items()))
        for arg, desc in sorted_parameters.items():
            param = key2param(arg)
            self.argmap[param] = arg

            if desc.get("pattern"):
                self.pattern_params[param] = desc["pattern"]
            if desc.get("enum"):
                self.enum_params[param] = desc["enum"]
            if desc.get("required"):
                self.required_params.append(param)
            if desc.get("repeated"):
                self.repeated_params.append(param)
            if desc.get("location") == "query":
                self.query_params.append(param)
            if desc.get("location") == "path":
                self.path_params.add(param)
            self.param_types[param] = desc.get("type", "string")

        # TODO(dhermes): Determine if this is still necessary. Discovery based APIs
        #                should have all path parameters already marked with
        #                'location: path'.
        for match in URITEMPLATE.finditer(method_desc["path"]):
            for namematch in VARNAME.finditer(match.group(0)):
                name = key2param(namematch.group(0))
                self.path_params.add(name)
                if name in self.query_params:
                    self.query_params.remove(name)


def createMethod(methodName, methodDesc, rootDesc, schema):
    """Creates a method for attaching to a Resource.

    Args:
      methodName: string, name of the method to use.
      methodDesc: object, fragment of deserialized discovery document that
        describes the method.
      rootDesc: object, the entire deserialized discovery document.
      schema: object, mapping of schema names to schema descriptions.
    """
    methodName = fix_method_name(methodName)
    (
        pathUrl,
        httpMethod,
        methodId,
        accept,
        maxSize,
        mediaPathUrl,
    ) = _fix_up_method_description(methodDesc, rootDesc, schema)

    parameters = ResourceMethodParameters(methodDesc)

    def method(self, **kwargs):
        # Don't bother with doc string, it will be over-written by createMethod.

        for name in kwargs:
            if name not in parameters.argmap:
                raise TypeError("Got an unexpected keyword argument {}".format(name))

        # Remove args that have a value of None.
        keys = list(kwargs.keys())
        for name in keys:
            if kwargs[name] is None:
                del kwargs[name]

        for name in parameters.required_params:
            if name not in kwargs:
                # temporary workaround for non-paging methods incorrectly requiring
                # page token parameter (cf. drive.changes.watch vs. drive.changes.list)
                if name not in _PAGE_TOKEN_NAMES or _findPageTokenName(
                    _methodProperties(methodDesc, schema, "response")
                ):
                    raise TypeError('Missing required parameter "%s"' % name)

        for name, regex in parameters.pattern_params.items():
            if name in kwargs:
                if isinstance(kwargs[name], str):
                    pvalues = [kwargs[name]]
                else:
                    pvalues = kwargs[name]
                for pvalue in pvalues:
                    if re.match(regex, pvalue) is None:
                        raise TypeError(
                            'Parameter "%s" value "%s" does not match the pattern "%s"'
                            % (name, pvalue, regex)
                        )

        for name, enums in parameters.enum_params.items():
            if name in kwargs:
                # We need to handle the case of a repeated enum
                # name differently, since we want to handle both
                # arg='value' and arg=['value1', 'value2']
                if name in parameters.repeated_params and not isinstance(
                    kwargs[name], str
                ):
                    values = kwargs[name]
                else:
                    values = [kwargs[name]]
                for value in values:
                    if value not in enums:
                        raise TypeError(
                            'Parameter "%s" value "%s" is not an allowed value in "%s"'
                            % (name, value, str(enums))
                        )

        actual_query_params = {}
        actual_path_params = {}
        for key, value in kwargs.items():
            to_type = parameters.param_types.get(key, "string")
            # For repeated parameters we cast each member of the list.
            if key in parameters.repeated_params and type(value) == type([]):
                cast_value = [_cast(x, to_type) for x in value]
            else:
                cast_value = _cast(value, to_type)
            if key in parameters.query_params:
                actual_query_params[parameters.argmap[key]] = cast_value
            if key in parameters.path_params:
                actual_path_params[parameters.argmap[key]] = cast_value
        body_value = kwargs.get("body", None)
        media_filename = kwargs.get("media_body", None)
        media_mime_type = kwargs.get("media_mime_type", None)

        if self._developerKey:
            actual_query_params["key"] = self._developerKey

        model = self._model
        if methodName.endswith("_media"):
            model = MediaModel()
        elif "response" not in methodDesc:
            model = RawModel()

        headers = {}
        headers, params, query, body = model.request(
            headers, actual_path_params, actual_query_params, body_value
        )

        expanded_url = uritemplate.expand(pathUrl, params)
        url = _urljoin(self._baseUrl, expanded_url + query)

        resumable = None
        multipart_boundary = ""

        if media_filename:
            # Ensure we end up with a valid MediaUpload object.
            if isinstance(media_filename, str):
                if media_mime_type is None:
                    logger.warning(
                        "media_mime_type argument not specified: trying to auto-detect for %s",
                        media_filename,
                    )
                    media_mime_type, _ = mimetypes.guess_type(media_filename)
                if media_mime_type is None:
                    raise UnknownFileType(media_filename)
                if not mimeparse.best_match([media_mime_type], ",".join(accept)):
                    raise UnacceptableMimeTypeError(media_mime_type)
                media_upload = MediaFileUpload(media_filename, mimetype=media_mime_type)
            elif isinstance(media_filename, MediaUpload):
                media_upload = media_filename
            else:
                raise TypeError("media_filename must be str or MediaUpload.")

            # Check the maxSize
            if media_upload.size() is not None and media_upload.size() > maxSize > 0:
                raise MediaUploadSizeError("Media larger than: %s" % maxSize)

            # Use the media path uri for media uploads
            expanded_url = uritemplate.expand(mediaPathUrl, params)
            url = _urljoin(self._baseUrl, expanded_url + query)
            url = _fix_up_media_path_base_url(url, self._baseUrl)
            if media_upload.resumable():
                url = _add_query_parameter(url, "uploadType", "resumable")

            if media_upload.resumable():
                # This is all we need to do for resumable, if the body exists it gets
                # sent in the first request, otherwise an empty body is sent.
                resumable = media_upload
            else:
                # A non-resumable upload
                if body is None:
                    # This is a simple media upload
                    headers["content-type"] = media_upload.mimetype()
                    body = media_upload.getbytes(0, media_upload.size())
                    url = _add_query_parameter(url, "uploadType", "media")
                else:
                    # This is a multipart/related upload.
                    msgRoot = MIMEMultipart("related")
                    # msgRoot should not write out it's own headers
                    setattr(msgRoot, "_write_headers", lambda self: None)

                    # attach the body as one part
                    msg = MIMENonMultipart(*headers["content-type"].split("/"))
                    msg.set_payload(body)
                    msgRoot.attach(msg)

                    # attach the media as the second part
                    msg = MIMENonMultipart(*media_upload.mimetype().split("/"))
                    msg["Content-Transfer-Encoding"] = "binary"

                    payload = media_upload.getbytes(0, media_upload.size())
                    msg.set_payload(payload)
                    msgRoot.attach(msg)
                    # encode the body: note that we can't use `as_string`, because
                    # it plays games with `From ` lines.
                    fp = io.BytesIO()
                    g = _BytesGenerator(fp, mangle_from_=False)
                    g.flatten(msgRoot, unixfrom=False)
                    body = fp.getvalue()

                    multipart_boundary = msgRoot.get_boundary()
                    headers["content-type"] = (
                        "multipart/related; " 'boundary="%s"'
                    ) % multipart_boundary
                    url = _add_query_parameter(url, "uploadType", "multipart")

        logger.debug("URL being requested: %s %s" % (httpMethod, url))
        return self._requestBuilder(
            self._http,
            model.response,
            url,
            method=httpMethod,
            body=body,
            headers=headers,
            methodId=methodId,
            resumable=resumable,
        )

    docs = [methodDesc.get("description", DEFAULT_METHOD_DOC), "\n\n"]
    if len(parameters.argmap) > 0:
        docs.append("Args:\n")

    # Skip undocumented params and params common to all methods.
    skip_parameters = list(rootDesc.get("parameters", {}).keys())
    skip_parameters.extend(STACK_QUERY_PARAMETERS)

    all_args = list(parameters.argmap.keys())
    args_ordered = [key2param(s) for s in methodDesc.get("parameterOrder", [])]

    # Move body to the front of the line.
    if "body" in all_args:
        args_ordered.append("body")

    for name in sorted(all_args):
        if name not in args_ordered:
            args_ordered.append(name)

    for arg in args_ordered:
        if arg in skip_parameters:
            continue

        repeated = ""
        if arg in parameters.repeated_params:
            repeated = " (repeated)"
        required = ""
        if arg in parameters.required_params:
            required = " (required)"
        paramdesc = methodDesc["parameters"][parameters.argmap[arg]]
        paramdoc = paramdesc.get("description", "A parameter")
        if "$ref" in paramdesc:
            docs.append(
                ("  %s: object, %s%s%s\n    The object takes the form of:\n\n%s\n\n")
                % (
                    arg,
                    paramdoc,
                    required,
                    repeated,
                    schema.prettyPrintByName(paramdesc["$ref"]),
                )
            )
        else:
            paramtype = paramdesc.get("type", "string")
            docs.append(
                "  %s: %s, %s%s%s\n" % (arg, paramtype, paramdoc, required, repeated)
            )
        enum = paramdesc.get("enum", [])
        enumDesc = paramdesc.get("enumDescriptions", [])
        if enum and enumDesc:
            docs.append("    Allowed values\n")
            for (name, desc) in zip(enum, enumDesc):
                docs.append("      %s - %s\n" % (name, desc))
    if "response" in methodDesc:
        if methodName.endswith("_media"):
            docs.append("\nReturns:\n  The media object as a string.\n\n    ")
        else:
            docs.append("\nReturns:\n  An object of the form:\n\n    ")
            docs.append(schema.prettyPrintSchema(methodDesc["response"]))

    setattr(method, "__doc__", "".join(docs))
    return (methodName, method)


def createNextMethod(
    methodName,
    pageTokenName="pageToken",
    nextPageTokenName="nextPageToken",
    isPageTokenParameter=True,
):
    """Creates any _next methods for attaching to a Resource.

    The _next methods allow for easy iteration through list() responses.

    Args:
      methodName: string, name of the method to use.
      pageTokenName: string, name of request page token field.
      nextPageTokenName: string, name of response page token field.
      isPageTokenParameter: Boolean, True if request page token is a query
          parameter, False if request page token is a field of the request body.
    """
    methodName = fix_method_name(methodName)

    def methodNext(self, previous_request, previous_response):
        """Retrieves the next page of results.

        Args:
          previous_request: The request for the previous page. (required)
          previous_response: The response from the request for the previous page. (required)

        Returns:
          A request object that you can call 'execute()' on to request the next
          page. Returns None if there are no more items in the collection.
        """
        # Retrieve nextPageToken from previous_response
        # Use as pageToken in previous_request to create new request.

        nextPageToken = previous_response.get(nextPageTokenName, None)
        if not nextPageToken:
            return None

        request = copy.copy(previous_request)

        if isPageTokenParameter:
            # Replace pageToken value in URI
            request.uri = _add_query_parameter(
                request.uri, pageTokenName, nextPageToken
            )
            logger.debug("Next page request URL: %s %s" % (methodName, request.uri))
        else:
            # Replace pageToken value in request body
            model = self._model
            body = model.deserialize(request.body)
            body[pageTokenName] = nextPageToken
            request.body = model.serialize(body)
            request.body_size = len(request.body)
            if "content-length" in request.headers:
                del request.headers["content-length"]
            logger.debug("Next page request body: %s %s" % (methodName, body))

        return request

    return (methodName, methodNext)


class Resource(object):
    """A class for interacting with a resource."""

    def __init__(
        self,
        http,
        baseUrl,
        model,
        requestBuilder,
        developerKey,
        resourceDesc,
        rootDesc,
        schema,
    ):
        """Build a Resource from the API description.

        Args:
          http: httplib2.Http, Object to make http requests with.
          baseUrl: string, base URL for the API. All requests are relative to this
              URI.
          model: googleapiclient.Model, converts to and from the wire format.
          requestBuilder: class or callable that instantiates an
              googleapiclient.HttpRequest object.
          developerKey: string, key obtained from
              https://code.google.com/apis/console
          resourceDesc: object, section of deserialized discovery document that
              describes a resource. Note that the top level discovery document
              is considered a resource.
          rootDesc: object, the entire deserialized discovery document.
          schema: object, mapping of schema names to schema descriptions.
        """
        self._dynamic_attrs = []

        self._http = http
        self._baseUrl = baseUrl
        self._model = model
        self._developerKey = developerKey
        self._requestBuilder = requestBuilder
        self._resourceDesc = resourceDesc
        self._rootDesc = rootDesc
        self._schema = schema

        self._set_service_methods()

    def _set_dynamic_attr(self, attr_name, value):
        """Sets an instance attribute and tracks it in a list of dynamic attributes.

        Args:
          attr_name: string; The name of the attribute to be set
          value: The value being set on the object and tracked in the dynamic cache.
        """
        self._dynamic_attrs.append(attr_name)
        self.__dict__[attr_name] = value

    def __getstate__(self):
        """Trim the state down to something that can be pickled.

        Uses the fact that the instance variable _dynamic_attrs holds attrs that
        will be wiped and restored on pickle serialization.
        """
        state_dict = copy.copy(self.__dict__)
        for dynamic_attr in self._dynamic_attrs:
            del state_dict[dynamic_attr]
        del state_dict["_dynamic_attrs"]
        return state_dict

    def __setstate__(self, state):
        """Reconstitute the state of the object from being pickled.

        Uses the fact that the instance variable _dynamic_attrs holds attrs that
        will be wiped and restored on pickle serialization.
        """
        self.__dict__.update(state)
        self._dynamic_attrs = []
        self._set_service_methods()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()

    def close(self):
        """Close httplib2 connections."""
        # httplib2 leaves sockets open by default.
        # Cleanup using the `close` method.
        # https://github.com/httplib2/httplib2/issues/148
        self._http.close()

    def _set_service_methods(self):
        self._add_basic_methods(self._resourceDesc, self._rootDesc, self._schema)
        self._add_nested_resources(self._resourceDesc, self._rootDesc, self._schema)
        self._add_next_methods(self._resourceDesc, self._schema)

    def _add_basic_methods(self, resourceDesc, rootDesc, schema):
        # If this is the root Resource, add a new_batch_http_request() method.
        if resourceDesc == rootDesc:
            batch_uri = "%s%s" % (
                rootDesc["rootUrl"],
                rootDesc.get("batchPath", "batch"),
            )

            def new_batch_http_request(callback=None):
                """Create a BatchHttpRequest object based on the discovery document.

                Args:
                  callback: callable, A callback to be called for each response, of the
                    form callback(id, response, exception). The first parameter is the
                    request id, and the second is the deserialized response object. The
                    third is an apiclient.errors.HttpError exception object if an HTTP
                    error occurred while processing the request, or None if no error
                    occurred.

                Returns:
                  A BatchHttpRequest object based on the discovery document.
                """
                return BatchHttpRequest(callback=callback, batch_uri=batch_uri)

            self._set_dynamic_attr("new_batch_http_request", new_batch_http_request)

        # Add basic methods to Resource
        if "methods" in resourceDesc:
            for methodName, methodDesc in resourceDesc["methods"].items():
                fixedMethodName, method = createMethod(
                    methodName, methodDesc, rootDesc, schema
                )
                self._set_dynamic_attr(
                    fixedMethodName, method.__get__(self, self.__class__)
                )
                # Add in _media methods. The functionality of the attached method will
                # change when it sees that the method name ends in _media.
                if methodDesc.get("supportsMediaDownload", False):
                    fixedMethodName, method = createMethod(
                        methodName + "_media", methodDesc, rootDesc, schema
                    )
                    self._set_dynamic_attr(
                        fixedMethodName, method.__get__(self, self.__class__)
                    )

    def _add_nested_resources(self, resourceDesc, rootDesc, schema):
        # Add in nested resources
        if "resources" in resourceDesc:

            def createResourceMethod(methodName, methodDesc):
                """Create a method on the Resource to access a nested Resource.

                Args:
                  methodName: string, name of the method to use.
                  methodDesc: object, fragment of deserialized discovery document that
                    describes the method.
                """
                methodName = fix_method_name(methodName)

                def methodResource(self):
                    return Resource(
                        http=self._http,
                        baseUrl=self._baseUrl,
                        model=self._model,
                        developerKey=self._developerKey,
                        requestBuilder=self._requestBuilder,
                        resourceDesc=methodDesc,
                        rootDesc=rootDesc,
                        schema=schema,
                    )

                setattr(methodResource, "__doc__", "A collection resource.")
                setattr(methodResource, "__is_resource__", True)

                return (methodName, methodResource)

            for methodName, methodDesc in resourceDesc["resources"].items():
                fixedMethodName, method = createResourceMethod(methodName, methodDesc)
                self._set_dynamic_attr(
                    fixedMethodName, method.__get__(self, self.__class__)
                )

    def _add_next_methods(self, resourceDesc, schema):
        # Add _next() methods if and only if one of the names 'pageToken' or
        # 'nextPageToken' occurs among the fields of both the method's response
        # type either the method's request (query parameters) or request body.
        if "methods" not in resourceDesc:
            return
        for methodName, methodDesc in resourceDesc["methods"].items():
            nextPageTokenName = _findPageTokenName(
                _methodProperties(methodDesc, schema, "response")
            )
            if not nextPageTokenName:
                continue
            isPageTokenParameter = True
            pageTokenName = _findPageTokenName(methodDesc.get("parameters", {}))
            if not pageTokenName:
                isPageTokenParameter = False
                pageTokenName = _findPageTokenName(
                    _methodProperties(methodDesc, schema, "request")
                )
            if not pageTokenName:
                continue
            fixedMethodName, method = createNextMethod(
                methodName + "_next",
                pageTokenName,
                nextPageTokenName,
                isPageTokenParameter,
            )
            self._set_dynamic_attr(
                fixedMethodName, method.__get__(self, self.__class__)
            )


def _findPageTokenName(fields):
    """Search field names for one like a page token.

    Args:
      fields: container of string, names of fields.

    Returns:
      First name that is either 'pageToken' or 'nextPageToken' if one exists,
      otherwise None.
    """
    return next(
        (tokenName for tokenName in _PAGE_TOKEN_NAMES if tokenName in fields), None
    )


def _methodProperties(methodDesc, schema, name):
    """Get properties of a field in a method description.

    Args:
      methodDesc: object, fragment of deserialized discovery document that
        describes the method.
      schema: object, mapping of schema names to schema descriptions.
      name: string, name of top-level field in method description.

    Returns:
      Object representing fragment of deserialized discovery document
      corresponding to 'properties' field of object corresponding to named field
      in method description, if it exists, otherwise empty dict.
    """
    desc = methodDesc.get(name, {})
    if "$ref" in desc:
        desc = schema.get(desc["$ref"], {})
    return desc.get("properties", {})
