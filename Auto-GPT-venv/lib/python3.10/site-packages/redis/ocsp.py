import base64
import datetime
import ssl
from urllib.parse import urljoin, urlparse

import cryptography.hazmat.primitives.hashes
import requests
from cryptography import hazmat, x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA1, Hash
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp

from redis.exceptions import AuthorizationError, ConnectionError


def _verify_response(issuer_cert, ocsp_response):
    pubkey = issuer_cert.public_key()
    try:
        if isinstance(pubkey, RSAPublicKey):
            pubkey.verify(
                ocsp_response.signature,
                ocsp_response.tbs_response_bytes,
                PKCS1v15(),
                ocsp_response.signature_hash_algorithm,
            )
        elif isinstance(pubkey, DSAPublicKey):
            pubkey.verify(
                ocsp_response.signature,
                ocsp_response.tbs_response_bytes,
                ocsp_response.signature_hash_algorithm,
            )
        elif isinstance(pubkey, EllipticCurvePublicKey):
            pubkey.verify(
                ocsp_response.signature,
                ocsp_response.tbs_response_bytes,
                ECDSA(ocsp_response.signature_hash_algorithm),
            )
        else:
            pubkey.verify(ocsp_response.signature, ocsp_response.tbs_response_bytes)
    except InvalidSignature:
        raise ConnectionError("failed to valid ocsp response")


def _check_certificate(issuer_cert, ocsp_bytes, validate=True):
    """A wrapper the return the validity of a known ocsp certificate"""

    ocsp_response = ocsp.load_der_ocsp_response(ocsp_bytes)

    if ocsp_response.response_status == ocsp.OCSPResponseStatus.UNAUTHORIZED:
        raise AuthorizationError("you are not authorized to view this ocsp certificate")
    if ocsp_response.response_status == ocsp.OCSPResponseStatus.SUCCESSFUL:
        if ocsp_response.certificate_status != ocsp.OCSPCertStatus.GOOD:
            raise ConnectionError(
                f'Received an {str(ocsp_response.certificate_status).split(".")[1]} '
                "ocsp certificate status"
            )
    else:
        raise ConnectionError(
            "failed to retrieve a sucessful response from the ocsp responder"
        )

    if ocsp_response.this_update >= datetime.datetime.now():
        raise ConnectionError("ocsp certificate was issued in the future")

    if (
        ocsp_response.next_update
        and ocsp_response.next_update < datetime.datetime.now()
    ):
        raise ConnectionError("ocsp certificate has invalid update - in the past")

    responder_name = ocsp_response.responder_name
    issuer_hash = ocsp_response.issuer_key_hash
    responder_hash = ocsp_response.responder_key_hash

    cert_to_validate = issuer_cert
    if (
        responder_name is not None
        and responder_name == issuer_cert.subject
        or responder_hash == issuer_hash
    ):
        cert_to_validate = issuer_cert
    else:
        certs = ocsp_response.certificates
        responder_certs = _get_certificates(
            certs, issuer_cert, responder_name, responder_hash
        )

        try:
            responder_cert = responder_certs[0]
        except IndexError:
            raise ConnectionError("no certificates found for the responder")

        ext = responder_cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
        if ext is None or x509.oid.ExtendedKeyUsageOID.OCSP_SIGNING not in ext.value:
            raise ConnectionError("delegate not autorized for ocsp signing")
        cert_to_validate = responder_cert

    if validate:
        _verify_response(cert_to_validate, ocsp_response)
    return True


def _get_certificates(certs, issuer_cert, responder_name, responder_hash):
    if responder_name is None:
        certificates = [
            c
            for c in certs
            if _get_pubkey_hash(c) == responder_hash and c.issuer == issuer_cert.subject
        ]
    else:
        certificates = [
            c
            for c in certs
            if c.subject == responder_name and c.issuer == issuer_cert.subject
        ]

    return certificates


def _get_pubkey_hash(certificate):
    pubkey = certificate.public_key()

    # https://stackoverflow.com/a/46309453/600498
    if isinstance(pubkey, RSAPublicKey):
        h = pubkey.public_bytes(Encoding.DER, PublicFormat.PKCS1)
    elif isinstance(pubkey, EllipticCurvePublicKey):
        h = pubkey.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    else:
        h = pubkey.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)

    sha1 = Hash(SHA1(), backend=backends.default_backend())
    sha1.update(h)
    return sha1.finalize()


def ocsp_staple_verifier(con, ocsp_bytes, expected=None):
    """An implemention of a function for set_ocsp_client_callback in PyOpenSSL.

    This function validates that the provide ocsp_bytes response is valid,
    and matches the expected, stapled responses.
    """
    if ocsp_bytes in [b"", None]:
        raise ConnectionError("no ocsp response present")

    issuer_cert = None
    peer_cert = con.get_peer_certificate().to_cryptography()
    for c in con.get_peer_cert_chain():
        cert = c.to_cryptography()
        if cert.subject == peer_cert.issuer:
            issuer_cert = cert
            break

    if issuer_cert is None:
        raise ConnectionError("no matching issuer cert found in certificate chain")

    if expected is not None:
        e = x509.load_pem_x509_certificate(expected)
        if peer_cert != e:
            raise ConnectionError("received and expected certificates do not match")

    return _check_certificate(issuer_cert, ocsp_bytes)


class OCSPVerifier:
    """A class to verify ssl sockets for RFC6960/RFC6961. This can be used
    when using direct validation of OCSP responses and certificate revocations.

    @see https://datatracker.ietf.org/doc/html/rfc6960
    @see https://datatracker.ietf.org/doc/html/rfc6961
    """

    def __init__(self, sock, host, port, ca_certs=None):
        self.SOCK = sock
        self.HOST = host
        self.PORT = port
        self.CA_CERTS = ca_certs

    def _bin2ascii(self, der):
        """Convert SSL certificates in a binary (DER) format to ASCII PEM."""

        pem = ssl.DER_cert_to_PEM_cert(der)
        cert = x509.load_pem_x509_certificate(pem.encode(), backends.default_backend())
        return cert

    def components_from_socket(self):
        """This function returns the certificate, primary issuer, and primary ocsp
        server in the chain for a socket already wrapped with ssl.
        """

        # convert the binary certifcate to text
        der = self.SOCK.getpeercert(True)
        if der is False:
            raise ConnectionError("no certificate found for ssl peer")
        cert = self._bin2ascii(der)
        return self._certificate_components(cert)

    def _certificate_components(self, cert):
        """Given an SSL certificate, retract the useful components for
        validating the certificate status with an OCSP server.

        Args:
            cert ([bytes]): A PEM encoded ssl certificate
        """

        try:
            aia = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.AUTHORITY_INFORMATION_ACCESS
            ).value
        except cryptography.x509.extensions.ExtensionNotFound:
            raise ConnectionError("No AIA information present in ssl certificate")

        # fetch certificate issuers
        issuers = [
            i
            for i in aia
            if i.access_method == x509.oid.AuthorityInformationAccessOID.CA_ISSUERS
        ]
        try:
            issuer = issuers[0].access_location.value
        except IndexError:
            issuer = None

        # now, the series of ocsp server entries
        ocsps = [
            i
            for i in aia
            if i.access_method == x509.oid.AuthorityInformationAccessOID.OCSP
        ]

        try:
            ocsp = ocsps[0].access_location.value
        except IndexError:
            raise ConnectionError("no ocsp servers in certificate")

        return cert, issuer, ocsp

    def components_from_direct_connection(self):
        """Return the certificate, primary issuer, and primary ocsp server
        from the host defined by the socket. This is useful in cases where
        different certificates are occasionally presented.
        """

        pem = ssl.get_server_certificate((self.HOST, self.PORT), ca_certs=self.CA_CERTS)
        cert = x509.load_pem_x509_certificate(pem.encode(), backends.default_backend())
        return self._certificate_components(cert)

    def build_certificate_url(self, server, cert, issuer_cert):
        """Return the complete url to the ocsp"""
        orb = ocsp.OCSPRequestBuilder()

        # add_certificate returns an initialized OCSPRequestBuilder
        orb = orb.add_certificate(
            cert, issuer_cert, cryptography.hazmat.primitives.hashes.SHA256()
        )
        request = orb.build()

        path = base64.b64encode(
            request.public_bytes(hazmat.primitives.serialization.Encoding.DER)
        )
        url = urljoin(server, path.decode("ascii"))
        return url

    def check_certificate(self, server, cert, issuer_url):
        """Checks the validitity of an ocsp server for an issuer"""

        r = requests.get(issuer_url)
        if not r.ok:
            raise ConnectionError("failed to fetch issuer certificate")
        der = r.content
        issuer_cert = self._bin2ascii(der)

        ocsp_url = self.build_certificate_url(server, cert, issuer_cert)

        # HTTP 1.1 mandates the addition of the Host header in ocsp responses
        header = {
            "Host": urlparse(ocsp_url).netloc,
            "Content-Type": "application/ocsp-request",
        }
        r = requests.get(ocsp_url, headers=header)
        if not r.ok:
            raise ConnectionError("failed to fetch ocsp certificate")
        return _check_certificate(issuer_cert, r.content, True)

    def is_valid(self):
        """Returns the validity of the certificate wrapping our socket.
        This first retrieves for validate the certificate, issuer_url,
        and ocsp_server for certificate validate. Then retrieves the
        issuer certificate from the issuer_url, and finally checks
        the validity of OCSP revocation status.
        """

        # validate the certificate
        try:
            cert, issuer_url, ocsp_server = self.components_from_socket()
            if issuer_url is None:
                raise ConnectionError("no issuers found in certificate chain")
            return self.check_certificate(ocsp_server, cert, issuer_url)
        except AuthorizationError:
            cert, issuer_url, ocsp_server = self.components_from_direct_connection()
            if issuer_url is None:
                raise ConnectionError("no issuers found in certificate chain")
            return self.check_certificate(ocsp_server, cert, issuer_url)
