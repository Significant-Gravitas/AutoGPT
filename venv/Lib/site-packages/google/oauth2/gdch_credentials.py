# Copyright 2022 Google LLC
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

"""Experimental GDCH credentials support.
"""

import datetime

from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
from google.oauth2 import _client


TOKEN_EXCHANGE_TYPE = "urn:ietf:params:oauth:token-type:token-exchange"
ACCESS_TOKEN_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
SERVICE_ACCOUNT_TOKEN_TYPE = "urn:k8s:params:oauth:token-type:serviceaccount"
JWT_LIFETIME = datetime.timedelta(seconds=3600)  # 1 hour


class ServiceAccountCredentials(credentials.Credentials):
    """Credentials for GDCH (`Google Distributed Cloud Hosted`_) for service
    account users.

    .. _Google Distributed Cloud Hosted:
        https://cloud.google.com/blog/topics/hybrid-cloud/\
            announcing-google-distributed-cloud-edge-and-hosted

    To create a GDCH service account credential, first create a JSON file of
    the following format::

        {
            "type": "gdch_service_account",
            "format_version": "1",
            "project": "<project name>",
            "private_key_id": "<key id>",
            "private_key": "-----BEGIN EC PRIVATE KEY-----\n<key bytes>\n-----END EC PRIVATE KEY-----\n",
            "name": "<service identity name>",
            "ca_cert_path": "<CA cert path>",
            "token_uri": "https://service-identity.<Domain>/authenticate"
        }

    The "format_version" field stands for the format of the JSON file. For now
    it is always "1". The `private_key_id` and `private_key` is used for signing.
    The `ca_cert_path` is used for token server TLS certificate verification.

    After the JSON file is created, set `GOOGLE_APPLICATION_CREDENTIALS` environment
    variable to the JSON file path, then use the following code to create the
    credential::

        import google.auth

        credential, _ = google.auth.default()
        credential = credential.with_gdch_audience("<the audience>")

    We can also create the credential directly::

        from google.oauth import gdch_credentials

        credential = gdch_credentials.ServiceAccountCredentials.from_service_account_file("<the json file path>")
        credential = credential.with_gdch_audience("<the audience>")

    The token is obtained in the following way. This class first creates a
    self signed JWT. It uses the `name` value as the `iss` and `sub` claim, and
    the `token_uri` as the `aud` claim, and signs the JWT with the `private_key`.
    It then sends the JWT to the `token_uri` to exchange a final token for
    `audience`.
    """

    def __init__(
        self, signer, service_identity_name, project, audience, token_uri, ca_cert_path
    ):
        """
        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            service_identity_name (str): The service identity name. It will be
                used as the `iss` and `sub` claim in the self signed JWT.
            project (str): The project.
            audience (str): The audience for the final token.
            token_uri (str): The token server uri.
            ca_cert_path (str): The CA cert path for token server side TLS
                certificate verification. If the token server uses well known
                CA, then this parameter can be `None`.
        """
        super(ServiceAccountCredentials, self).__init__()
        self._signer = signer
        self._service_identity_name = service_identity_name
        self._project = project
        self._audience = audience
        self._token_uri = token_uri
        self._ca_cert_path = ca_cert_path

    def _create_jwt(self):
        now = _helpers.utcnow()
        expiry = now + JWT_LIFETIME
        iss_sub_value = "system:serviceaccount:{}:{}".format(
            self._project, self._service_identity_name
        )

        payload = {
            "iss": iss_sub_value,
            "sub": iss_sub_value,
            "aud": self._token_uri,
            "iat": _helpers.datetime_to_secs(now),
            "exp": _helpers.datetime_to_secs(expiry),
        }

        return _helpers.from_bytes(jwt.encode(self._signer, payload))

    @_helpers.copy_docstring(credentials.Credentials)
    def refresh(self, request):
        import google.auth.transport.requests

        if not isinstance(request, google.auth.transport.requests.Request):
            raise exceptions.RefreshError(
                "For GDCH service account credentials, request must be a google.auth.transport.requests.Request object"
            )

        # Create a self signed JWT, and do token exchange.
        jwt_token = self._create_jwt()
        request_body = {
            "grant_type": TOKEN_EXCHANGE_TYPE,
            "audience": self._audience,
            "requested_token_type": ACCESS_TOKEN_TOKEN_TYPE,
            "subject_token": jwt_token,
            "subject_token_type": SERVICE_ACCOUNT_TOKEN_TYPE,
        }
        response_data = _client._token_endpoint_request(
            request,
            self._token_uri,
            request_body,
            access_token=None,
            use_json=True,
            verify=self._ca_cert_path,
        )

        self.token, _, self.expiry, _ = _client._handle_refresh_grant_response(
            response_data, None
        )

    def with_gdch_audience(self, audience):
        """Create a copy of GDCH credentials with the specified audience.

        Args:
            audience (str): The intended audience for GDCH credentials.
        """
        return self.__class__(
            self._signer,
            self._service_identity_name,
            self._project,
            audience,
            self._token_uri,
            self._ca_cert_path,
        )

    @classmethod
    def _from_signer_and_info(cls, signer, info):
        """Creates a Credentials instance from a signer and service account
        info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.

        Returns:
            google.oauth2.gdch_credentials.ServiceAccountCredentials: The constructed
                credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        if info["format_version"] != "1":
            raise ValueError("Only format version 1 is supported")

        return cls(
            signer,
            info["name"],  # service_identity_name
            info["project"],
            None,  # audience
            info["token_uri"],
            info.get("ca_cert_path", None),
        )

    @classmethod
    def from_service_account_info(cls, info):
        """Creates a Credentials instance from parsed service account info.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.oauth2.gdch_credentials.ServiceAccountCredentials: The constructed
                credentials.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        signer = _service_account_info.from_dict(
            info,
            require=[
                "format_version",
                "private_key_id",
                "private_key",
                "name",
                "project",
                "token_uri",
            ],
            use_rsa_signer=False,
        )
        return cls._from_signer_and_info(signer, info)

    @classmethod
    def from_service_account_file(cls, filename):
        """Creates a Credentials instance from a service account json file.

        Args:
            filename (str): The path to the service account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.oauth2.gdch_credentials.ServiceAccountCredentials: The constructed
                credentials.
        """
        info, signer = _service_account_info.from_filename(
            filename,
            require=[
                "format_version",
                "private_key_id",
                "private_key",
                "name",
                "project",
                "token_uri",
            ],
            use_rsa_signer=False,
        )
        return cls._from_signer_and_info(signer, info)
