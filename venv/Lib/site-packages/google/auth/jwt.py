# Copyright 2016 Google LLC
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
    from google.auth import jwt

    signer = crypt.Signer(private_key)
    payload = {'some': 'payload'}
    encoded = jwt.encode(signer, payload)

To decode a JWT and verify claims use :func:`decode`::

    claims = jwt.decode(encoded, certs=public_certs)

You can also skip verification::

    claims = jwt.decode(encoded, verify=False)

.. _rfc7519: https://tools.ietf.org/html/rfc7519

"""

try:
    from collections.abc import Mapping
# Python 2.7 compatibility
except ImportError:  # pragma: NO COVER
    from collections import Mapping  # type: ignore
import copy
import datetime
import json

import cachetools
import six
from six.moves import urllib

from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials

try:
    from google.auth.crypt import es256
except ImportError:  # pragma: NO COVER
    es256 = None  # type: ignore

_DEFAULT_TOKEN_LIFETIME_SECS = 3600  # 1 hour in seconds
_DEFAULT_MAX_CACHE_SIZE = 10
_ALGORITHM_TO_VERIFIER_CLASS = {"RS256": crypt.RSAVerifier}
_CRYPTOGRAPHY_BASED_ALGORITHMS = frozenset(["ES256"])

if es256 is not None:  # pragma: NO COVER
    _ALGORITHM_TO_VERIFIER_CLASS["ES256"] = es256.ES256Verifier  # type: ignore


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
    if header is None:
        header = {}

    if key_id is None:
        key_id = signer.key_id

    header.update({"typ": "JWT"})

    if "alg" not in header:
        if es256 is not None and isinstance(signer, es256.ES256Signer):
            header.update({"alg": "ES256"})
        else:
            header.update({"alg": "RS256"})

    if key_id is not None:
        header["kid"] = key_id

    segments = [
        _helpers.unpadded_urlsafe_b64encode(json.dumps(header).encode("utf-8")),
        _helpers.unpadded_urlsafe_b64encode(json.dumps(payload).encode("utf-8")),
    ]

    signing_input = b".".join(segments)
    signature = signer.sign(signing_input)
    segments.append(_helpers.unpadded_urlsafe_b64encode(signature))

    return b".".join(segments)


def _decode_jwt_segment(encoded_section):
    """Decodes a single JWT segment."""
    section_bytes = _helpers.padded_urlsafe_b64decode(encoded_section)
    try:
        return json.loads(section_bytes.decode("utf-8"))
    except ValueError as caught_exc:
        new_exc = exceptions.MalformedError(
            "Can't parse segment: {0}".format(section_bytes)
        )
        six.raise_from(new_exc, caught_exc)


def _unverified_decode(token):
    """Decodes a token and does no verification.

    Args:
        token (Union[str, bytes]): The encoded JWT.

    Returns:
        Tuple[Mapping, Mapping, str, str]: header, payload, signed_section, and
            signature.

    Raises:
        google.auth.exceptions.MalformedError: if there are an incorrect amount of segments in the token or segments of the wrong type.
    """
    token = _helpers.to_bytes(token)

    if token.count(b".") != 2:
        raise exceptions.MalformedError(
            "Wrong number of segments in token: {0}".format(token)
        )

    encoded_header, encoded_payload, signature = token.split(b".")
    signed_section = encoded_header + b"." + encoded_payload
    signature = _helpers.padded_urlsafe_b64decode(signature)

    # Parse segments
    header = _decode_jwt_segment(encoded_header)
    payload = _decode_jwt_segment(encoded_payload)

    if not isinstance(header, Mapping):
        raise exceptions.MalformedError(
            "Header segment should be a JSON object: {0}".format(encoded_header)
        )

    if not isinstance(payload, Mapping):
        raise exceptions.MalformedError(
            "Payload segment should be a JSON object: {0}".format(encoded_payload)
        )

    return header, payload, signed_section, signature


def decode_header(token):
    """Return the decoded header of a token.

    No verification is done. This is useful to extract the key id from
    the header in order to acquire the appropriate certificate to verify
    the token.

    Args:
        token (Union[str, bytes]): the encoded JWT.

    Returns:
        Mapping: The decoded JWT header.
    """
    header, _, _, _ = _unverified_decode(token)
    return header


def _verify_iat_and_exp(payload, clock_skew_in_seconds=0):
    """Verifies the ``iat`` (Issued At) and ``exp`` (Expires) claims in a token
    payload.

    Args:
        payload (Mapping[str, str]): The JWT payload.
        clock_skew_in_seconds (int): The clock skew used for `iat` and `exp`
            validation.

    Raises:
        google.auth.exceptions.InvalidValue: if value validation failed.
        google.auth.exceptions.MalformedError: if schema validation failed.
    """
    now = _helpers.datetime_to_secs(_helpers.utcnow())

    # Make sure the iat and exp claims are present.
    for key in ("iat", "exp"):
        if key not in payload:
            raise exceptions.MalformedError(
                "Token does not contain required claim {}".format(key)
            )

    # Make sure the token wasn't issued in the future.
    iat = payload["iat"]
    # Err on the side of accepting a token that is slightly early to account
    # for clock skew.
    earliest = iat - clock_skew_in_seconds
    if now < earliest:
        raise exceptions.InvalidValue(
            "Token used too early, {} < {}. Check that your computer's clock is set correctly.".format(
                now, iat
            )
        )

    # Make sure the token wasn't issued in the past.
    exp = payload["exp"]
    # Err on the side of accepting a token that is slightly out of date
    # to account for clow skew.
    latest = exp + clock_skew_in_seconds
    if latest < now:
        raise exceptions.InvalidValue("Token expired, {} < {}".format(latest, now))


def decode(token, certs=None, verify=True, audience=None, clock_skew_in_seconds=0):
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
        audience (str or list): The audience claim, 'aud', that this JWT should
            contain. Or a list of audience claims. If None then the JWT's 'aud'
            parameter is not verified.
        clock_skew_in_seconds (int): The clock skew used for `iat` and `exp`
            validation.

    Returns:
        Mapping[str, str]: The deserialized JSON payload in the JWT.

    Raises:
        google.auth.exceptions.InvalidValue: if value validation failed.
        google.auth.exceptions.MalformedError: if schema validation failed.
    """
    header, payload, signed_section, signature = _unverified_decode(token)

    if not verify:
        return payload

    # Pluck the key id and algorithm from the header and make sure we have
    # a verifier that can support it.
    key_alg = header.get("alg")
    key_id = header.get("kid")

    try:
        verifier_cls = _ALGORITHM_TO_VERIFIER_CLASS[key_alg]
    except KeyError as exc:
        if key_alg in _CRYPTOGRAPHY_BASED_ALGORITHMS:
            six.raise_from(
                exceptions.InvalidValue(
                    "The key algorithm {} requires the cryptography package "
                    "to be installed.".format(key_alg)
                ),
                exc,
            )
        else:
            six.raise_from(
                exceptions.InvalidValue(
                    "Unsupported signature algorithm {}".format(key_alg)
                ),
                exc,
            )

    # If certs is specified as a dictionary of key IDs to certificates, then
    # use the certificate identified by the key ID in the token header.
    if isinstance(certs, Mapping):
        if key_id:
            if key_id not in certs:
                raise exceptions.MalformedError(
                    "Certificate for key id {} not found.".format(key_id)
                )
            certs_to_check = [certs[key_id]]
        # If there's no key id in the header, check against all of the certs.
        else:
            certs_to_check = certs.values()
    else:
        certs_to_check = certs

    # Verify that the signature matches the message.
    if not crypt.verify_signature(
        signed_section, signature, certs_to_check, verifier_cls
    ):
        raise exceptions.MalformedError("Could not verify token signature.")

    # Verify the issued at and created times in the payload.
    _verify_iat_and_exp(payload, clock_skew_in_seconds)

    # Check audience.
    if audience is not None:
        claim_audience = payload.get("aud")
        if isinstance(audience, str):
            audience = [audience]
        if claim_audience not in audience:
            raise exceptions.InvalidValue(
                "Token has wrong audience {}, expected one of {}".format(
                    claim_audience, audience
                )
            )

    return payload


class Credentials(
    google.auth.credentials.Signing, google.auth.credentials.CredentialsWithQuotaProject
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
        credentials = jwt.Credentials.from_service_account_file(
            'service-account.json',
            audience=audience)

    If you already have the service account file loaded and parsed::

        service_account_info = json.load(open('service_account.json'))
        credentials = jwt.Credentials.from_service_account_info(
            service_account_info,
            audience=audience)

    Both helper methods pass on arguments to the constructor, so you can
    specify the JWT claims::

        credentials = jwt.Credentials.from_service_account_file(
            'service-account.json',
            audience=audience,
            additional_claims={'meta': 'data'})

    You can also construct the credentials directly if you have a
    :class:`~google.auth.crypt.Signer` instance::

        credentials = jwt.Credentials(
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

    def __init__(
        self,
        signer,
        issuer,
        subject,
        audience,
        additional_claims=None,
        token_lifetime=_DEFAULT_TOKEN_LIFETIME_SECS,
        quota_project_id=None,
    ):
        """
        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            issuer (str): The `iss` claim.
            subject (str): The `sub` claim.
            audience (str): the `aud` claim. The intended audience for the
                credentials.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload.
            token_lifetime (int): The amount of time in seconds for
                which the token is valid. Defaults to 1 hour.
            quota_project_id (Optional[str]): The project ID used for quota
                and billing.
        """
        super(Credentials, self).__init__()
        self._signer = signer
        self._issuer = issuer
        self._subject = subject
        self._audience = audience
        self._token_lifetime = token_lifetime
        self._quota_project_id = quota_project_id

        if additional_claims is None:
            additional_claims = {}

        self._additional_claims = additional_claims

    @classmethod
    def _from_signer_and_info(cls, signer, info, **kwargs):
        """Creates a Credentials instance from a signer and service account
        info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
        kwargs.setdefault("subject", info["client_email"])
        kwargs.setdefault("issuer", info["client_email"])
        return cls(signer, **kwargs)

    @classmethod
    def from_service_account_info(cls, info, **kwargs):
        """Creates an Credentials instance from a dictionary.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
        signer = _service_account_info.from_dict(info, require=["client_email"])
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename, **kwargs):
        """Creates a Credentials instance from a service account .json file
        in Google format.

        Args:
            filename (str): The path to the service account .json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: The constructed credentials.
        """
        info, signer = _service_account_info.from_filename(
            filename, require=["client_email"]
        )
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_signing_credentials(cls, credentials, audience, **kwargs):
        """Creates a new :class:`google.auth.jwt.Credentials` instance from an
        existing :class:`google.auth.credentials.Signing` instance.

        The new instance will use the same signer as the existing instance and
        will use the existing instance's signer email as the issuer and
        subject by default.

        Example::

            svc_creds = service_account.Credentials.from_service_account_file(
                'service_account.json')
            audience = (
                'https://pubsub.googleapis.com/google.pubsub.v1.Publisher')
            jwt_creds = jwt.Credentials.from_signing_credentials(
                svc_creds, audience=audience)

        Args:
            credentials (google.auth.credentials.Signing): The credentials to
                use to construct the new credentials.
            audience (str): the `aud` claim. The intended audience for the
                credentials.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: A new Credentials instance.
        """
        kwargs.setdefault("issuer", credentials.signer_email)
        kwargs.setdefault("subject", credentials.signer_email)
        return cls(credentials.signer, audience=audience, **kwargs)

    def with_claims(
        self, issuer=None, subject=None, audience=None, additional_claims=None
    ):
        """Returns a copy of these credentials with modified claims.

        Args:
            issuer (str): The `iss` claim. If unspecified the current issuer
                claim will be used.
            subject (str): The `sub` claim. If unspecified the current subject
                claim will be used.
            audience (str): the `aud` claim. If unspecified the current
                audience claim will be used.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload. This will be merged with the current
                additional claims.

        Returns:
            google.auth.jwt.Credentials: A new credentials instance.
        """
        new_additional_claims = copy.deepcopy(self._additional_claims)
        new_additional_claims.update(additional_claims or {})

        return self.__class__(
            self._signer,
            issuer=issuer if issuer is not None else self._issuer,
            subject=subject if subject is not None else self._subject,
            audience=audience if audience is not None else self._audience,
            additional_claims=new_additional_claims,
            quota_project_id=self._quota_project_id,
        )

    @_helpers.copy_docstring(google.auth.credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):
        return self.__class__(
            self._signer,
            issuer=self._issuer,
            subject=self._subject,
            audience=self._audience,
            additional_claims=self._additional_claims,
            quota_project_id=quota_project_id,
        )

    def _make_jwt(self):
        """Make a signed JWT.

        Returns:
            Tuple[bytes, datetime]: The encoded JWT and the expiration.
        """
        now = _helpers.utcnow()
        lifetime = datetime.timedelta(seconds=self._token_lifetime)
        expiry = now + lifetime

        payload = {
            "iss": self._issuer,
            "sub": self._subject,
            "iat": _helpers.datetime_to_secs(now),
            "exp": _helpers.datetime_to_secs(expiry),
        }
        if self._audience:
            payload["aud"] = self._audience

        payload.update(self._additional_claims)

        jwt = encode(self._signer, payload)

        return jwt, expiry

    def refresh(self, request):
        """Refreshes the access token.

        Args:
            request (Any): Unused.
        """
        # pylint: disable=unused-argument
        # (pylint doesn't correctly recognize overridden methods.)
        self.token, self.expiry = self._make_jwt()

    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def sign_bytes(self, message):
        return self._signer.sign(message)

    @property  # type: ignore
    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def signer_email(self):
        return self._issuer

    @property  # type: ignore
    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def signer(self):
        return self._signer

    @property  # type: ignore
    def additional_claims(self):
        """ Additional claims the JWT object was created with."""
        return self._additional_claims


class OnDemandCredentials(
    google.auth.credentials.Signing, google.auth.credentials.CredentialsWithQuotaProject
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

    def __init__(
        self,
        signer,
        issuer,
        subject,
        additional_claims=None,
        token_lifetime=_DEFAULT_TOKEN_LIFETIME_SECS,
        max_cache_size=_DEFAULT_MAX_CACHE_SIZE,
        quota_project_id=None,
    ):
        """
        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            issuer (str): The `iss` claim.
            subject (str): The `sub` claim.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload.
            token_lifetime (int): The amount of time in seconds for
                which the token is valid. Defaults to 1 hour.
            max_cache_size (int): The maximum number of JWT tokens to keep in
                cache. Tokens are cached using :class:`cachetools.LRUCache`.
            quota_project_id (Optional[str]): The project ID used for quota
                and billing.

        """
        super(OnDemandCredentials, self).__init__()
        self._signer = signer
        self._issuer = issuer
        self._subject = subject
        self._token_lifetime = token_lifetime
        self._quota_project_id = quota_project_id

        if additional_claims is None:
            additional_claims = {}

        self._additional_claims = additional_claims
        self._cache = cachetools.LRUCache(maxsize=max_cache_size)

    @classmethod
    def _from_signer_and_info(cls, signer, info, **kwargs):
        """Creates an OnDemandCredentials instance from a signer and service
        account info.

        Args:
            signer (google.auth.crypt.Signer): The signer used to sign JWTs.
            info (Mapping[str, str]): The service account info.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
        kwargs.setdefault("subject", info["client_email"])
        kwargs.setdefault("issuer", info["client_email"])
        return cls(signer, **kwargs)

    @classmethod
    def from_service_account_info(cls, info, **kwargs):
        """Creates an OnDemandCredentials instance from a dictionary.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.

        Raises:
            google.auth.exceptions.MalformedError: If the info is not in the expected format.
        """
        signer = _service_account_info.from_dict(info, require=["client_email"])
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename, **kwargs):
        """Creates an OnDemandCredentials instance from a service account .json
        file in Google format.

        Args:
            filename (str): The path to the service account .json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.OnDemandCredentials: The constructed credentials.
        """
        info, signer = _service_account_info.from_filename(
            filename, require=["client_email"]
        )
        return cls._from_signer_and_info(signer, info, **kwargs)

    @classmethod
    def from_signing_credentials(cls, credentials, **kwargs):
        """Creates a new :class:`google.auth.jwt.OnDemandCredentials` instance
        from an existing :class:`google.auth.credentials.Signing` instance.

        The new instance will use the same signer as the existing instance and
        will use the existing instance's signer email as the issuer and
        subject by default.

        Example::

            svc_creds = service_account.Credentials.from_service_account_file(
                'service_account.json')
            jwt_creds = jwt.OnDemandCredentials.from_signing_credentials(
                svc_creds)

        Args:
            credentials (google.auth.credentials.Signing): The credentials to
                use to construct the new credentials.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.jwt.Credentials: A new Credentials instance.
        """
        kwargs.setdefault("issuer", credentials.signer_email)
        kwargs.setdefault("subject", credentials.signer_email)
        return cls(credentials.signer, **kwargs)

    def with_claims(self, issuer=None, subject=None, additional_claims=None):
        """Returns a copy of these credentials with modified claims.

        Args:
            issuer (str): The `iss` claim. If unspecified the current issuer
                claim will be used.
            subject (str): The `sub` claim. If unspecified the current subject
                claim will be used.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload. This will be merged with the current
                additional claims.

        Returns:
            google.auth.jwt.OnDemandCredentials: A new credentials instance.
        """
        new_additional_claims = copy.deepcopy(self._additional_claims)
        new_additional_claims.update(additional_claims or {})

        return self.__class__(
            self._signer,
            issuer=issuer if issuer is not None else self._issuer,
            subject=subject if subject is not None else self._subject,
            additional_claims=new_additional_claims,
            max_cache_size=self._cache.maxsize,
            quota_project_id=self._quota_project_id,
        )

    @_helpers.copy_docstring(google.auth.credentials.CredentialsWithQuotaProject)
    def with_quota_project(self, quota_project_id):

        return self.__class__(
            self._signer,
            issuer=self._issuer,
            subject=self._subject,
            additional_claims=self._additional_claims,
            max_cache_size=self._cache.maxsize,
            quota_project_id=quota_project_id,
        )

    @property
    def valid(self):
        """Checks the validity of the credentials.

        These credentials are always valid because it generates tokens on
        demand.
        """
        return True

    def _make_jwt_for_audience(self, audience):
        """Make a new JWT for the given audience.

        Args:
            audience (str): The intended audience.

        Returns:
            Tuple[bytes, datetime]: The encoded JWT and the expiration.
        """
        now = _helpers.utcnow()
        lifetime = datetime.timedelta(seconds=self._token_lifetime)
        expiry = now + lifetime

        payload = {
            "iss": self._issuer,
            "sub": self._subject,
            "iat": _helpers.datetime_to_secs(now),
            "exp": _helpers.datetime_to_secs(expiry),
            "aud": audience,
        }

        payload.update(self._additional_claims)

        jwt = encode(self._signer, payload)

        return jwt, expiry

    def _get_jwt_for_audience(self, audience):
        """Get a JWT For a given audience.

        If there is already an existing, non-expired token in the cache for
        the audience, that token is used. Otherwise, a new token will be
        created.

        Args:
            audience (str): The intended audience.

        Returns:
            bytes: The encoded JWT.
        """
        token, expiry = self._cache.get(audience, (None, None))

        if token is None or expiry < _helpers.utcnow():
            token, expiry = self._make_jwt_for_audience(audience)
            self._cache[audience] = token, expiry

        return token

    def refresh(self, request):
        """Raises an exception, these credentials can not be directly
        refreshed.

        Args:
            request (Any): Unused.

        Raises:
            google.auth.RefreshError
        """
        # pylint: disable=unused-argument
        # (pylint doesn't correctly recognize overridden methods.)
        raise exceptions.RefreshError(
            "OnDemandCredentials can not be directly refreshed."
        )

    def before_request(self, request, method, url, headers):
        """Performs credential-specific before request logic.

        Args:
            request (Any): Unused. JWT credentials do not need to make an
                HTTP request to refresh.
            method (str): The request's HTTP method.
            url (str): The request's URI. This is used as the audience claim
                when generating the JWT.
            headers (Mapping): The request's headers.
        """
        # pylint: disable=unused-argument
        # (pylint doesn't correctly recognize overridden methods.)
        parts = urllib.parse.urlsplit(url)
        # Strip query string and fragment
        audience = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, parts.path, "", "")
        )
        token = self._get_jwt_for_audience(audience)
        self.apply(headers, token=token)

    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def sign_bytes(self, message):
        return self._signer.sign(message)

    @property  # type: ignore
    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def signer_email(self):
        return self._issuer

    @property  # type: ignore
    @_helpers.copy_docstring(google.auth.credentials.Signing)
    def signer(self):
        return self._signer
