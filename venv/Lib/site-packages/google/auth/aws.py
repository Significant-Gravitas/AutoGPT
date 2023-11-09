# Copyright 2020 Google LLC
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

"""AWS Credentials and AWS Signature V4 Request Signer.

This module provides credentials to access Google Cloud resources from Amazon
Web Services (AWS) workloads. These credentials are recommended over the
use of service account credentials in AWS as they do not involve the management
of long-live service account private keys.

AWS Credentials are initialized using external_account arguments which are
typically loaded from the external credentials JSON file.
Unlike other Credentials that can be initialized with a list of explicit
arguments, secrets or credentials, external account clients use the
environment and hints/guidelines provided by the external_account JSON
file to retrieve credentials and exchange them for Google access tokens.

This module also provides a basic implementation of the
`AWS Signature Version 4`_ request signing algorithm.

AWS Credentials use serialized signed requests to the
`AWS STS GetCallerIdentity`_ API that can be exchanged for Google access tokens
via the GCP STS endpoint.

.. _AWS Signature Version 4: https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
.. _AWS STS GetCallerIdentity: https://docs.aws.amazon.com/STS/latest/APIReference/API_GetCallerIdentity.html
"""

import hashlib
import hmac
import json
import os
import posixpath
import re

from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin

from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account

# AWS Signature Version 4 signing algorithm identifier.
_AWS_ALGORITHM = "AWS4-HMAC-SHA256"
# The termination string for the AWS credential scope value as defined in
# https://docs.aws.amazon.com/general/latest/gr/sigv4-create-string-to-sign.html
_AWS_REQUEST_TYPE = "aws4_request"
# The AWS authorization header name for the security session token if available.
_AWS_SECURITY_TOKEN_HEADER = "x-amz-security-token"
# The AWS authorization header name for the auto-generated date.
_AWS_DATE_HEADER = "x-amz-date"


class RequestSigner(object):
    """Implements an AWS request signer based on the AWS Signature Version 4 signing
    process.
    https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
    """

    def __init__(self, region_name):
        """Instantiates an AWS request signer used to compute authenticated signed
        requests to AWS APIs based on the AWS Signature Version 4 signing process.

        Args:
            region_name (str): The AWS region to use.
        """

        self._region_name = region_name

    def get_request_options(
        self,
        aws_security_credentials,
        url,
        method,
        request_payload="",
        additional_headers={},
    ):
        """Generates the signed request for the provided HTTP request for calling
        an AWS API. This follows the steps described at:
        https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html

        Args:
            aws_security_credentials (Mapping[str, str]): A dictionary containing
                the AWS security credentials.
            url (str): The AWS service URL containing the canonical URI and
                query string.
            method (str): The HTTP method used to call this API.
            request_payload (Optional[str]): The optional request payload if
                available.
            additional_headers (Optional[Mapping[str, str]]): The optional
                additional headers needed for the requested AWS API.

        Returns:
            Mapping[str, str]: The AWS signed request dictionary object.
        """
        # Get AWS credentials.
        access_key = aws_security_credentials.get("access_key_id")
        secret_key = aws_security_credentials.get("secret_access_key")
        security_token = aws_security_credentials.get("security_token")

        additional_headers = additional_headers or {}

        uri = urllib.parse.urlparse(url)
        # Normalize the URL path. This is needed for the canonical_uri.
        # os.path.normpath can't be used since it normalizes "/" paths
        # to "\\" in Windows OS.
        normalized_uri = urllib.parse.urlparse(
            urljoin(url, posixpath.normpath(uri.path))
        )
        # Validate provided URL.
        if not uri.hostname or uri.scheme != "https":
            raise exceptions.InvalidResource("Invalid AWS service URL")

        header_map = _generate_authentication_header_map(
            host=uri.hostname,
            canonical_uri=normalized_uri.path or "/",
            canonical_querystring=_get_canonical_querystring(uri.query),
            method=method,
            region=self._region_name,
            access_key=access_key,
            secret_key=secret_key,
            security_token=security_token,
            request_payload=request_payload,
            additional_headers=additional_headers,
        )
        headers = {
            "Authorization": header_map.get("authorization_header"),
            "host": uri.hostname,
        }
        # Add x-amz-date if available.
        if "amz_date" in header_map:
            headers[_AWS_DATE_HEADER] = header_map.get("amz_date")
        # Append additional optional headers, eg. X-Amz-Target, Content-Type, etc.
        for key in additional_headers:
            headers[key] = additional_headers[key]

        # Add session token if available.
        if security_token is not None:
            headers[_AWS_SECURITY_TOKEN_HEADER] = security_token

        signed_request = {"url": url, "method": method, "headers": headers}
        if request_payload:
            signed_request["data"] = request_payload
        return signed_request


def _get_canonical_querystring(query):
    """Generates the canonical query string given a raw query string.
    Logic is based on
    https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html

    Args:
        query (str): The raw query string.

    Returns:
        str: The canonical query string.
    """
    # Parse raw query string.
    querystring = urllib.parse.parse_qs(query)
    querystring_encoded_map = {}
    for key in querystring:
        quote_key = urllib.parse.quote(key, safe="-_.~")
        # URI encode key.
        querystring_encoded_map[quote_key] = []
        for item in querystring[key]:
            # For each key, URI encode all values for that key.
            querystring_encoded_map[quote_key].append(
                urllib.parse.quote(item, safe="-_.~")
            )
        # Sort values for each key.
        querystring_encoded_map[quote_key].sort()
    # Sort keys.
    sorted_keys = list(querystring_encoded_map.keys())
    sorted_keys.sort()
    # Reconstruct the query string. Preserve keys with multiple values.
    querystring_encoded_pairs = []
    for key in sorted_keys:
        for item in querystring_encoded_map[key]:
            querystring_encoded_pairs.append("{}={}".format(key, item))
    return "&".join(querystring_encoded_pairs)


def _sign(key, msg):
    """Creates the HMAC-SHA256 hash of the provided message using the provided
    key.

    Args:
        key (str): The HMAC-SHA256 key to use.
        msg (str): The message to hash.

    Returns:
        str: The computed hash bytes.
    """
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signing_key(key, date_stamp, region_name, service_name):
    """Calculates the signing key used to calculate the signature for
    AWS Signature Version 4 based on:
    https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html

    Args:
        key (str): The AWS secret access key.
        date_stamp (str): The '%Y%m%d' date format.
        region_name (str): The AWS region.
        service_name (str): The AWS service name, eg. sts.

    Returns:
        str: The signing key bytes.
    """
    k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _generate_authentication_header_map(
    host,
    canonical_uri,
    canonical_querystring,
    method,
    region,
    access_key,
    secret_key,
    security_token,
    request_payload="",
    additional_headers={},
):
    """Generates the authentication header map needed for generating the AWS
    Signature Version 4 signed request.

    Args:
        host (str): The AWS service URL hostname.
        canonical_uri (str): The AWS service URL path name.
        canonical_querystring (str): The AWS service URL query string.
        method (str): The HTTP method used to call this API.
        region (str): The AWS region.
        access_key (str): The AWS access key ID.
        secret_key (str): The AWS secret access key.
        security_token (Optional[str]): The AWS security session token. This is
            available for temporary sessions.
        request_payload (Optional[str]): The optional request payload if
            available.
        additional_headers (Optional[Mapping[str, str]]): The optional
            additional headers needed for the requested AWS API.

    Returns:
        Mapping[str, str]: The AWS authentication header dictionary object.
            This contains the x-amz-date and authorization header information.
    """
    # iam.amazonaws.com host => iam service.
    # sts.us-east-2.amazonaws.com host => sts service.
    service_name = host.split(".")[0]

    current_time = _helpers.utcnow()
    amz_date = current_time.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = current_time.strftime("%Y%m%d")

    # Change all additional headers to be lower case.
    full_headers = {}
    for key in additional_headers:
        full_headers[key.lower()] = additional_headers[key]
    # Add AWS session token if available.
    if security_token is not None:
        full_headers[_AWS_SECURITY_TOKEN_HEADER] = security_token

    # Required headers
    full_headers["host"] = host
    # Do not use generated x-amz-date if the date header is provided.
    # Previously the date was not fixed with x-amz- and could be provided
    # manually.
    # https://github.com/boto/botocore/blob/879f8440a4e9ace5d3cf145ce8b3d5e5ffb892ef/tests/unit/auth/aws4_testsuite/get-header-value-trim.req
    if "date" not in full_headers:
        full_headers[_AWS_DATE_HEADER] = amz_date

    # Header keys need to be sorted alphabetically.
    canonical_headers = ""
    header_keys = list(full_headers.keys())
    header_keys.sort()
    for key in header_keys:
        canonical_headers = "{}{}:{}\n".format(
            canonical_headers, key, full_headers[key]
        )
    signed_headers = ";".join(header_keys)

    payload_hash = hashlib.sha256((request_payload or "").encode("utf-8")).hexdigest()

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
    canonical_request = "{}\n{}\n{}\n{}\n{}\n{}".format(
        method,
        canonical_uri,
        canonical_querystring,
        canonical_headers,
        signed_headers,
        payload_hash,
    )

    credential_scope = "{}/{}/{}/{}".format(
        date_stamp, region, service_name, _AWS_REQUEST_TYPE
    )

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-create-string-to-sign.html
    string_to_sign = "{}\n{}\n{}\n{}".format(
        _AWS_ALGORITHM,
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    )

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
    signing_key = _get_signing_key(secret_key, date_stamp, region, service_name)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-add-signature-to-request.html
    authorization_header = "{} Credential={}/{}, SignedHeaders={}, Signature={}".format(
        _AWS_ALGORITHM, access_key, credential_scope, signed_headers, signature
    )

    authentication_header = {"authorization_header": authorization_header}
    # Do not use generated x-amz-date if the date header is provided.
    if "date" not in full_headers:
        authentication_header["amz_date"] = amz_date
    return authentication_header


class Credentials(external_account.Credentials):
    """AWS external account credentials.
    This is used to exchange serialized AWS signature v4 signed requests to
    AWS STS GetCallerIdentity service for Google access tokens.
    """

    def __init__(
        self,
        audience,
        subject_token_type,
        token_url,
        credential_source=None,
        *args,
        **kwargs
    ):
        """Instantiates an AWS workload external account credentials object.

        Args:
            audience (str): The STS audience field.
            subject_token_type (str): The subject token type.
            token_url (str): The STS endpoint URL.
            credential_source (Mapping): The credential source dictionary used
                to provide instructions on how to retrieve external credential
                to be exchanged for Google access tokens.
            args (List): Optional positional arguments passed into the underlying :meth:`~external_account.Credentials.__init__` method.
            kwargs (Mapping): Optional keyword arguments passed into the underlying :meth:`~external_account.Credentials.__init__` method.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                access token retrieval logic.
            ValueError: For invalid parameters.

        .. note:: Typically one of the helper constructors
            :meth:`from_file` or
            :meth:`from_info` are used instead of calling the constructor directly.
        """
        super(Credentials, self).__init__(
            audience=audience,
            subject_token_type=subject_token_type,
            token_url=token_url,
            credential_source=credential_source,
            *args,
            **kwargs
        )
        credential_source = credential_source or {}
        self._environment_id = credential_source.get("environment_id") or ""
        self._region_url = credential_source.get("region_url")
        self._security_credentials_url = credential_source.get("url")
        self._cred_verification_url = credential_source.get(
            "regional_cred_verification_url"
        )
        self._imdsv2_session_token_url = credential_source.get(
            "imdsv2_session_token_url"
        )
        self._region = None
        self._request_signer = None
        self._target_resource = audience

        # Get the environment ID. Currently, only one version supported (v1).
        matches = re.match(r"^(aws)([\d]+)$", self._environment_id)
        if matches:
            env_id, env_version = matches.groups()
        else:
            env_id, env_version = (None, None)

        if env_id != "aws" or self._cred_verification_url is None:
            raise exceptions.InvalidResource(
                "No valid AWS 'credential_source' provided"
            )
        elif int(env_version or "") != 1:
            raise exceptions.InvalidValue(
                "aws version '{}' is not supported in the current build.".format(
                    env_version
                )
            )

    def retrieve_subject_token(self, request):
        """Retrieves the subject token using the credential_source object.
        The subject token is a serialized `AWS GetCallerIdentity signed request`_.

        The logic is summarized as:

        Retrieve the AWS region from the AWS_REGION or AWS_DEFAULT_REGION
        environment variable or from the AWS metadata server availability-zone
        if not found in the environment variable.

        Check AWS credentials in environment variables. If not found, retrieve
        from the AWS metadata server security-credentials endpoint.

        When retrieving AWS credentials from the metadata server
        security-credentials endpoint, the AWS role needs to be determined by
        calling the security-credentials endpoint without any argument. Then the
        credentials can be retrieved via: security-credentials/role_name

        Generate the signed request to AWS STS GetCallerIdentity action.

        Inject x-goog-cloud-target-resource into header and serialize the
        signed request. This will be the subject-token to pass to GCP STS.

        .. _AWS GetCallerIdentity signed request:
            https://cloud.google.com/iam/docs/access-resources-aws#exchange-token

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
        Returns:
            str: The retrieved subject token.
        """
        # Fetch the session token required to make meta data endpoint calls to aws.
        if (
            request is not None
            and self._imdsv2_session_token_url is not None
            and self._should_use_metadata_server()
        ):
            headers = {"X-aws-ec2-metadata-token-ttl-seconds": "300"}

            imdsv2_session_token_response = request(
                url=self._imdsv2_session_token_url, method="PUT", headers=headers
            )

            if imdsv2_session_token_response.status != 200:
                raise exceptions.RefreshError(
                    "Unable to retrieve AWS Session Token",
                    imdsv2_session_token_response.data,
                )

            imdsv2_session_token = imdsv2_session_token_response.data
        else:
            imdsv2_session_token = None

        # Initialize the request signer if not yet initialized after determining
        # the current AWS region.
        if self._request_signer is None:
            self._region = self._get_region(
                request, self._region_url, imdsv2_session_token
            )
            self._request_signer = RequestSigner(self._region)

        # Retrieve the AWS security credentials needed to generate the signed
        # request.
        aws_security_credentials = self._get_security_credentials(
            request, imdsv2_session_token
        )
        # Generate the signed request to AWS STS GetCallerIdentity API.
        # Use the required regional endpoint. Otherwise, the request will fail.
        request_options = self._request_signer.get_request_options(
            aws_security_credentials,
            self._cred_verification_url.replace("{region}", self._region),
            "POST",
        )
        # The GCP STS endpoint expects the headers to be formatted as:
        # [
        #   {key: 'x-amz-date', value: '...'},
        #   {key: 'Authorization', value: '...'},
        #   ...
        # ]
        # And then serialized as:
        # quote(json.dumps({
        #   url: '...',
        #   method: 'POST',
        #   headers: [{key: 'x-amz-date', value: '...'}, ...]
        # }))
        request_headers = request_options.get("headers")
        # The full, canonical resource name of the workload identity pool
        # provider, with or without the HTTPS prefix.
        # Including this header as part of the signature is recommended to
        # ensure data integrity.
        request_headers["x-goog-cloud-target-resource"] = self._target_resource

        # Serialize AWS signed request.
        # Keeping inner keys in sorted order makes testing easier for Python
        # versions <=3.5 as the stringified JSON string would have a predictable
        # key order.
        aws_signed_req = {}
        aws_signed_req["url"] = request_options.get("url")
        aws_signed_req["method"] = request_options.get("method")
        aws_signed_req["headers"] = []
        # Reformat header to GCP STS expected format.
        for key in sorted(request_headers.keys()):
            aws_signed_req["headers"].append(
                {"key": key, "value": request_headers[key]}
            )

        return urllib.parse.quote(
            json.dumps(aws_signed_req, separators=(",", ":"), sort_keys=True)
        )

    def _get_region(self, request, url, imdsv2_session_token):
        """Retrieves the current AWS region from either the AWS_REGION or
        AWS_DEFAULT_REGION environment variable or from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            url (str): The AWS metadata server region URL.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            str: The current AWS region.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS region.
        """
        # The AWS metadata server is not available in some AWS environments
        # such as AWS lambda. Instead, it is available via environment
        # variable.
        env_aws_region = os.environ.get(environment_vars.AWS_REGION)
        if env_aws_region is not None:
            return env_aws_region

        env_aws_region = os.environ.get(environment_vars.AWS_DEFAULT_REGION)
        if env_aws_region is not None:
            return env_aws_region

        if not self._region_url:
            raise exceptions.RefreshError("Unable to determine AWS region")

        headers = None
        if imdsv2_session_token is not None:
            headers = {"X-aws-ec2-metadata-token": imdsv2_session_token}

        response = request(url=self._region_url, method="GET", headers=headers)

        # Support both string and bytes type response.data.
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        if response.status != 200:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS region", response_body
            )

        # This endpoint will return the region in format: us-east-2b.
        # Only the us-east-2 part should be used.
        return response_body[:-1]

    def _get_security_credentials(self, request, imdsv2_session_token):
        """Retrieves the AWS security credentials required for signing AWS
        requests from either the AWS security credentials environment variables
        or from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            Mapping[str, str]: The AWS security credentials dictionary object.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS security credentials.
        """

        # Check environment variables for permanent credentials first.
        # https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html
        env_aws_access_key_id = os.environ.get(environment_vars.AWS_ACCESS_KEY_ID)
        env_aws_secret_access_key = os.environ.get(
            environment_vars.AWS_SECRET_ACCESS_KEY
        )
        # This is normally not available for permanent credentials.
        env_aws_session_token = os.environ.get(environment_vars.AWS_SESSION_TOKEN)
        if env_aws_access_key_id and env_aws_secret_access_key:
            return {
                "access_key_id": env_aws_access_key_id,
                "secret_access_key": env_aws_secret_access_key,
                "security_token": env_aws_session_token,
            }

        # Get role name.
        role_name = self._get_metadata_role_name(request, imdsv2_session_token)

        # Get security credentials.
        credentials = self._get_metadata_security_credentials(
            request, role_name, imdsv2_session_token
        )

        return {
            "access_key_id": credentials.get("AccessKeyId"),
            "secret_access_key": credentials.get("SecretAccessKey"),
            "security_token": credentials.get("Token"),
        }

    def _get_metadata_security_credentials(
        self, request, role_name, imdsv2_session_token
    ):
        """Retrieves the AWS security credentials required for signing AWS
        requests from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            role_name (str): The AWS role name required by the AWS metadata
                server security_credentials endpoint in order to return the
                credentials.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            Mapping[str, str]: The AWS metadata server security credentials
                response.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS security credentials.
        """
        headers = {"Content-Type": "application/json"}
        if imdsv2_session_token is not None:
            headers["X-aws-ec2-metadata-token"] = imdsv2_session_token

        response = request(
            url="{}/{}".format(self._security_credentials_url, role_name),
            method="GET",
            headers=headers,
        )

        # support both string and bytes type response.data
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        if response.status != http_client.OK:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS security credentials", response_body
            )

        credentials_response = json.loads(response_body)

        return credentials_response

    def _get_metadata_role_name(self, request, imdsv2_session_token):
        """Retrieves the AWS role currently attached to the current AWS
        workload by querying the AWS metadata server. This is needed for the
        AWS metadata server security credentials endpoint in order to retrieve
        the AWS security credentials needed to sign requests to AWS APIs.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            str: The AWS role name.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS role name.
        """
        if self._security_credentials_url is None:
            raise exceptions.RefreshError(
                "Unable to determine the AWS metadata server security credentials endpoint"
            )

        headers = None
        if imdsv2_session_token is not None:
            headers = {"X-aws-ec2-metadata-token": imdsv2_session_token}

        response = request(
            url=self._security_credentials_url, method="GET", headers=headers
        )

        # support both string and bytes type response.data
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        if response.status != http_client.OK:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS role name", response_body
            )

        return response_body

    def _should_use_metadata_server(self):
        # The AWS region can be provided through AWS_REGION or AWS_DEFAULT_REGION.
        # The metadata server should be used if it cannot be retrieved from one of
        # these environment variables.
        if not os.environ.get(environment_vars.AWS_REGION) and not os.environ.get(
            environment_vars.AWS_DEFAULT_REGION
        ):
            return True

        # AWS security credentials can be retrieved from the AWS_ACCESS_KEY_ID
        # and AWS_SECRET_ACCESS_KEY environment variables. The metadata server
        # should be used if either of these are not available.
        if not os.environ.get(environment_vars.AWS_ACCESS_KEY_ID) or not os.environ.get(
            environment_vars.AWS_SECRET_ACCESS_KEY
        ):
            return True

        return False

    @classmethod
    def from_info(cls, info, **kwargs):
        """Creates an AWS Credentials instance from parsed external account info.

        Args:
            info (Mapping[str, str]): The AWS external account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.aws.Credentials: The constructed credentials.

        Raises:
            ValueError: For invalid parameters.
        """
        return super(Credentials, cls).from_info(info, **kwargs)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Creates an AWS Credentials instance from an external account json file.

        Args:
            filename (str): The path to the AWS external account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.aws.Credentials: The constructed credentials.
        """
        return super(Credentials, cls).from_file(filename, **kwargs)
