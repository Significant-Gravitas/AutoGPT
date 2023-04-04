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

"""JSON Web Tokens

Provides support for creating (encoding) and verifying (decoding) JWTs,
especially JWTs generated and consumed by Google infrastructure.

See `rfc7519`_ for more details on JWTs.

To encode a JWT use :func:`encode`::

    from google.auth import crypt
    from google.auth import jwt_async

    signer = crypt.Signer(private_key)
    payload = {'some': 'payload'}
    encoded = jwt_async.encode(signer, payload)

To decode a JWT and verify claims use :func:`decode`::

    claims = jwt_async.decode(encoded, certs=public_certs)

You can also skip verification::

    claims = jwt_async.decode(encoded, verify=False)

.. _rfc7519: https://tools.ietf.org/html/rfc7519


NOTE: This async support is experimental and marked internal. This surface may
change in minor releases.
"""

from google.auth import _credentials_async
from google.auth import jwt


def encode(signer, payload, header=None, key_id=None):
    """Make a signed JWT.

    Args:
        signer (google.auth.crypt.Signer): The signer used to sign the JWT.
        payload (Mapping[str, str]): The JWT payload.
        header (Mapping[str, str]): Additional JWT header payload.
        key_id (str): The key id to add to the JWT header. If the
            signer has a key id it will be used as the default. If this is
            specified it will override the signer's key id.

    Returns:
        bytes: The encoded JWT.
    """
    return jwt.encode(signer, payload, header, key_id)


def decode(token, certs=None, verify=True, audience=None):
    """Decode and verify a JWT.

    Args:
        token (str): The encoded JWT.
        certs (Union[str, bytes, Mapping[str, Union[str, bytes]]]): The
            certificate used to validate the JWT signature. If bytes or string,
            it must the the public key certificate in PEM format. If a mapping,
            it must be a mapping of key IDs to public key certificates in PEM
            format. The mapping must contain the same key ID that's specified
            in the token's header.
        verify (bool): Whether to perform signature and claim validation.
            Verification is done by default.
        audience (str): The audience claim, 'aud', that this JWT should
            contain. If None then the JWT's 'aud' parameter is not verified.

    Returns:
        Mapping[str, str]: The deserialized JSON payload in the JWT.

    Raises:
        ValueError: if any verification checks failed.
    """

    return jwt.decode(token, certs, verify, audience)


class Credentials(
    jwt.Credentials, _credentials_async.Signing, _credentials_async.Credentials
):
    """Credentials that use a JWT as the bearer token.

    These credentials require an "audience" claim. This claim identifies the
    intended recipient of the bearer token.

    The constructor arguments determine the claims for the JWT that is
    sent with requests. Usually, you'll construct these credentials with
    one of the helper constructors as shown in the next section.

    To create JWT credentials using a Google service account private key
    JSON file::

        audience = 'https://pubsub.googleapis.com/google.pubsub.v1.Publisher'
        credentials = jwt_async.Credentials.from_service_account_file(
            'service-account.json',
            audience=audience)

    If you already have the service account file loaded and parsed::

        service_account_info = json.load(open('service_account.json'))
        credentials = jwt_async.Credentials.from_service_account_info(
            service_account_info,
            audience=audience)

    Both helper methods pass on arguments to the constructor, so you can
    specify the JWT claims::

        credentials = jwt_async.Credentials.from_service_account_file(
            'service-account.json',
            audience=audience,
            additional_claims={'meta': 'data'})

    You can also construct the credentials directly if you have a
    :class:`~google.auth.crypt.Signer` instance::

        credentials = jwt_async.Credentials(
            signer,
            issuer='your-issuer',
            subject='your-subject',
            audience=audience)

    The claims are considered immutable. If you want to modify the claims,
    you can easily create another instance using :meth:`with_claims`::

        new_audience = (
            'https://pubsub.googleapis.com/google.pubsub.v1.Subscriber')
        new_credentials = credentials.with_claims(audience=new_audience)
    """


class OnDemandCredentials(
    jwt.OnDemandCredentials, _credentials_async.Signing, _credentials_async.Credentials
):
    """On-demand JWT credentials.

    Like :class:`Credentials`, this class uses a JWT as the bearer token for
    authentication. However, this class does not require the audience at
    construction time. Instead, it will generate a new token on-demand for
    each request using the request URI as the audience. It caches tokens
    so that multiple requests to the same URI do not incur the overhead
    of generating a new token every time.

    This behavior is especially useful for `gRPC`_ clients. A gRPC service may
    have multiple audience and gRPC clients may not know all of the audiences
    required for accessing a particular service. With these credentials,
    no knowledge of the audiences is required ahead of time.

    .. _grpc: http://www.grpc.io/
    """
