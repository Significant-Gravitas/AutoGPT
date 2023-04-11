# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2017 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Common DNSSEC-related functions and constants."""

from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union

import hashlib
import math
import struct
import time
import base64
from datetime import datetime

from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash

import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
import dns.rrset
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag


class UnsupportedAlgorithm(dns.exception.DNSException):
    """The DNSSEC algorithm is not supported."""


class AlgorithmKeyMismatch(UnsupportedAlgorithm):
    """The DNSSEC algorithm is not supported for the given key type."""


class ValidationFailure(dns.exception.DNSException):
    """The DNSSEC signature is invalid."""


class DeniedByPolicy(dns.exception.DNSException):
    """Denied by DNSSEC policy."""


PublicKey = Union[
    "rsa.RSAPublicKey",
    "ec.EllipticCurvePublicKey",
    "ed25519.Ed25519PublicKey",
    "ed448.Ed448PublicKey",
]

PrivateKey = Union[
    "rsa.RSAPrivateKey",
    "ec.EllipticCurvePrivateKey",
    "ed25519.Ed25519PrivateKey",
    "ed448.Ed448PrivateKey",
]


def algorithm_from_text(text: str) -> Algorithm:
    """Convert text into a DNSSEC algorithm value.

    *text*, a ``str``, the text to convert to into an algorithm value.

    Returns an ``int``.
    """

    return Algorithm.from_text(text)


def algorithm_to_text(value: Union[Algorithm, int]) -> str:
    """Convert a DNSSEC algorithm value to text

    *value*, a ``dns.dnssec.Algorithm``.

    Returns a ``str``, the name of a DNSSEC algorithm.
    """

    return Algorithm.to_text(value)


def to_timestamp(value: Union[datetime, str, float, int]) -> int:
    """Convert various format to a timestamp"""
    if isinstance(value, datetime):
        return int(value.timestamp())
    elif isinstance(value, str):
        return sigtime_to_posixtime(value)
    elif isinstance(value, float):
        return int(value)
    elif isinstance(value, int):
        return value
    else:
        raise TypeError("Unsupported timestamp type")


def key_id(key: Union[DNSKEY, CDNSKEY]) -> int:
    """Return the key id (a 16-bit number) for the specified key.

    *key*, a ``dns.rdtypes.ANY.DNSKEY.DNSKEY``

    Returns an ``int`` between 0 and 65535
    """

    rdata = key.to_wire()
    if key.algorithm == Algorithm.RSAMD5:
        return (rdata[-3] << 8) + rdata[-2]
    else:
        total = 0
        for i in range(len(rdata) // 2):
            total += (rdata[2 * i] << 8) + rdata[2 * i + 1]
        if len(rdata) % 2 != 0:
            total += rdata[len(rdata) - 1] << 8
        total += (total >> 16) & 0xFFFF
        return total & 0xFFFF


class Policy:
    def __init__(self):
        pass

    def ok_to_sign(self, _: DNSKEY) -> bool:  # pragma: no cover
        return False

    def ok_to_validate(self, _: DNSKEY) -> bool:  # pragma: no cover
        return False

    def ok_to_create_ds(self, _: DSDigest) -> bool:  # pragma: no cover
        return False

    def ok_to_validate_ds(self, _: DSDigest) -> bool:  # pragma: no cover
        return False


class SimpleDeny(Policy):
    def __init__(self, deny_sign, deny_validate, deny_create_ds, deny_validate_ds):
        super().__init__()
        self._deny_sign = deny_sign
        self._deny_validate = deny_validate
        self._deny_create_ds = deny_create_ds
        self._deny_validate_ds = deny_validate_ds

    def ok_to_sign(self, key: DNSKEY) -> bool:
        return key.algorithm not in self._deny_sign

    def ok_to_validate(self, key: DNSKEY) -> bool:
        return key.algorithm not in self._deny_validate

    def ok_to_create_ds(self, algorithm: DSDigest) -> bool:
        return algorithm not in self._deny_create_ds

    def ok_to_validate_ds(self, algorithm: DSDigest) -> bool:
        return algorithm not in self._deny_validate_ds


rfc_8624_policy = SimpleDeny(
    {Algorithm.RSAMD5, Algorithm.DSA, Algorithm.DSANSEC3SHA1, Algorithm.ECCGOST},
    {Algorithm.RSAMD5, Algorithm.DSA, Algorithm.DSANSEC3SHA1},
    {DSDigest.NULL, DSDigest.SHA1, DSDigest.GOST},
    {DSDigest.NULL},
)

allow_all_policy = SimpleDeny(set(), set(), set(), set())


default_policy = rfc_8624_policy


def make_ds(
    name: Union[dns.name.Name, str],
    key: dns.rdata.Rdata,
    algorithm: Union[DSDigest, str],
    origin: Optional[dns.name.Name] = None,
    policy: Optional[Policy] = None,
    validating: bool = False,
) -> DS:
    """Create a DS record for a DNSSEC key.

    *name*, a ``dns.name.Name`` or ``str``, the owner name of the DS record.

    *key*, a ``dns.rdtypes.ANY.DNSKEY.DNSKEY`` or ``dns.rdtypes.ANY.DNSKEY.CDNSKEY``,
    the key the DS is about.

    *algorithm*, a ``str`` or ``int`` specifying the hash algorithm.
    The currently supported hashes are "SHA1", "SHA256", and "SHA384". Case
    does not matter for these strings.

    *origin*, a ``dns.name.Name`` or ``None``.  If *key* is a relative name,
    then it will be made absolute using the specified origin.

    *policy*, a ``dns.dnssec.Policy`` or ``None``.  If ``None``, the default policy,
    ``dns.dnssec.default_policy`` is used; this policy defaults to that of RFC 8624.

    *validating*, a ``bool``.  If ``True``, then policy is checked in
    validating mode, i.e. "Is it ok to validate using this digest algorithm?".
    Otherwise the policy is checked in creating mode, i.e. "Is it ok to create a DS with
    this digest algorithm?".

    Raises ``UnsupportedAlgorithm`` if the algorithm is unknown.

    Raises ``DeniedByPolicy`` if the algorithm is denied by policy.

    Returns a ``dns.rdtypes.ANY.DS.DS``
    """

    if policy is None:
        policy = default_policy
    try:
        if isinstance(algorithm, str):
            algorithm = DSDigest[algorithm.upper()]
    except Exception:
        raise UnsupportedAlgorithm('unsupported algorithm "%s"' % algorithm)
    if validating:
        check = policy.ok_to_validate_ds
    else:
        check = policy.ok_to_create_ds
    if not check(algorithm):
        raise DeniedByPolicy
    if not isinstance(key, (DNSKEY, CDNSKEY)):
        raise ValueError("key is not a DNSKEY/CDNSKEY")
    if algorithm == DSDigest.SHA1:
        dshash = hashlib.sha1()
    elif algorithm == DSDigest.SHA256:
        dshash = hashlib.sha256()
    elif algorithm == DSDigest.SHA384:
        dshash = hashlib.sha384()
    else:
        raise UnsupportedAlgorithm('unsupported algorithm "%s"' % algorithm)

    if isinstance(name, str):
        name = dns.name.from_text(name, origin)
    wire = name.canonicalize().to_wire()
    assert wire is not None
    dshash.update(wire)
    dshash.update(key.to_wire(origin=origin))
    digest = dshash.digest()

    dsrdata = struct.pack("!HBB", key_id(key), key.algorithm, algorithm) + digest
    ds = dns.rdata.from_wire(
        dns.rdataclass.IN, dns.rdatatype.DS, dsrdata, 0, len(dsrdata)
    )
    return cast(DS, ds)


def make_cds(
    name: Union[dns.name.Name, str],
    key: dns.rdata.Rdata,
    algorithm: Union[DSDigest, str],
    origin: Optional[dns.name.Name] = None,
) -> CDS:
    """Create a CDS record for a DNSSEC key.

    *name*, a ``dns.name.Name`` or ``str``, the owner name of the DS record.

    *key*, a ``dns.rdtypes.ANY.DNSKEY.DNSKEY`` or ``dns.rdtypes.ANY.DNSKEY.CDNSKEY``,
    the key the DS is about.

    *algorithm*, a ``str`` or ``int`` specifying the hash algorithm.
    The currently supported hashes are "SHA1", "SHA256", and "SHA384". Case
    does not matter for these strings.

    *origin*, a ``dns.name.Name`` or ``None``.  If *key* is a relative name,
    then it will be made absolute using the specified origin.

    Raises ``UnsupportedAlgorithm`` if the algorithm is unknown.

    Returns a ``dns.rdtypes.ANY.DS.CDS``
    """

    ds = make_ds(name, key, algorithm, origin)
    return CDS(
        rdclass=ds.rdclass,
        rdtype=dns.rdatatype.CDS,
        key_tag=ds.key_tag,
        algorithm=ds.algorithm,
        digest_type=ds.digest_type,
        digest=ds.digest,
    )


def _find_candidate_keys(
    keys: Dict[dns.name.Name, Union[dns.rdataset.Rdataset, dns.node.Node]], rrsig: RRSIG
) -> Optional[List[DNSKEY]]:
    value = keys.get(rrsig.signer)
    if isinstance(value, dns.node.Node):
        rdataset = value.get_rdataset(dns.rdataclass.IN, dns.rdatatype.DNSKEY)
    else:
        rdataset = value
    if rdataset is None:
        return None
    return [
        cast(DNSKEY, rd)
        for rd in rdataset
        if rd.algorithm == rrsig.algorithm and key_id(rd) == rrsig.key_tag
    ]


def _is_rsa(algorithm: int) -> bool:
    return algorithm in (
        Algorithm.RSAMD5,
        Algorithm.RSASHA1,
        Algorithm.RSASHA1NSEC3SHA1,
        Algorithm.RSASHA256,
        Algorithm.RSASHA512,
    )


def _is_dsa(algorithm: int) -> bool:
    return algorithm in (Algorithm.DSA, Algorithm.DSANSEC3SHA1)


def _is_ecdsa(algorithm: int) -> bool:
    return algorithm in (Algorithm.ECDSAP256SHA256, Algorithm.ECDSAP384SHA384)


def _is_eddsa(algorithm: int) -> bool:
    return algorithm in (Algorithm.ED25519, Algorithm.ED448)


def _is_gost(algorithm: int) -> bool:
    return algorithm == Algorithm.ECCGOST


def _is_md5(algorithm: int) -> bool:
    return algorithm == Algorithm.RSAMD5


def _is_sha1(algorithm: int) -> bool:
    return algorithm in (
        Algorithm.DSA,
        Algorithm.RSASHA1,
        Algorithm.DSANSEC3SHA1,
        Algorithm.RSASHA1NSEC3SHA1,
    )


def _is_sha256(algorithm: int) -> bool:
    return algorithm in (Algorithm.RSASHA256, Algorithm.ECDSAP256SHA256)


def _is_sha384(algorithm: int) -> bool:
    return algorithm == Algorithm.ECDSAP384SHA384


def _is_sha512(algorithm: int) -> bool:
    return algorithm == Algorithm.RSASHA512


def _ensure_algorithm_key_combination(algorithm: int, key: PublicKey) -> None:
    """Ensure algorithm is valid for key type, throwing an exception on
    mismatch."""
    if isinstance(key, rsa.RSAPublicKey):
        if _is_rsa(algorithm):
            return
        raise AlgorithmKeyMismatch('algorithm "%s" not valid for RSA key' % algorithm)
    if isinstance(key, dsa.DSAPublicKey):
        if _is_dsa(algorithm):
            return
        raise AlgorithmKeyMismatch('algorithm "%s" not valid for DSA key' % algorithm)
    if isinstance(key, ec.EllipticCurvePublicKey):
        if _is_ecdsa(algorithm):
            return
        raise AlgorithmKeyMismatch('algorithm "%s" not valid for ECDSA key' % algorithm)
    if isinstance(key, ed25519.Ed25519PublicKey):
        if algorithm == Algorithm.ED25519:
            return
        raise AlgorithmKeyMismatch(
            'algorithm "%s" not valid for ED25519 key' % algorithm
        )
    if isinstance(key, ed448.Ed448PublicKey):
        if algorithm == Algorithm.ED448:
            return
        raise AlgorithmKeyMismatch('algorithm "%s" not valid for ED448 key' % algorithm)

    raise TypeError("unsupported key type")


def _make_hash(algorithm: int) -> Any:
    if _is_md5(algorithm):
        return hashes.MD5()
    if _is_sha1(algorithm):
        return hashes.SHA1()
    if _is_sha256(algorithm):
        return hashes.SHA256()
    if _is_sha384(algorithm):
        return hashes.SHA384()
    if _is_sha512(algorithm):
        return hashes.SHA512()
    if algorithm == Algorithm.ED25519:
        return hashes.SHA512()
    if algorithm == Algorithm.ED448:
        return hashes.SHAKE256(114)

    raise ValidationFailure("unknown hash for algorithm %u" % algorithm)


def _bytes_to_long(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _get_rrname_rdataset(
    rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
) -> Tuple[dns.name.Name, dns.rdataset.Rdataset]:
    if isinstance(rrset, tuple):
        return rrset[0], rrset[1]
    else:
        return rrset.name, rrset


def _validate_signature(sig: bytes, data: bytes, key: DNSKEY, chosen_hash: Any) -> None:
    keyptr: bytes
    if _is_rsa(key.algorithm):
        # we ignore because mypy is confused and thinks key.key is a str for unknown
        # reasons.
        keyptr = key.key
        (bytes_,) = struct.unpack("!B", keyptr[0:1])
        keyptr = keyptr[1:]
        if bytes_ == 0:
            (bytes_,) = struct.unpack("!H", keyptr[0:2])
            keyptr = keyptr[2:]
        rsa_e = keyptr[0:bytes_]
        rsa_n = keyptr[bytes_:]
        try:
            rsa_public_key = rsa.RSAPublicNumbers(
                _bytes_to_long(rsa_e), _bytes_to_long(rsa_n)
            ).public_key(default_backend())
        except ValueError:
            raise ValidationFailure("invalid public key")
        rsa_public_key.verify(sig, data, padding.PKCS1v15(), chosen_hash)
    elif _is_dsa(key.algorithm):
        keyptr = key.key
        (t,) = struct.unpack("!B", keyptr[0:1])
        keyptr = keyptr[1:]
        octets = 64 + t * 8
        dsa_q = keyptr[0:20]
        keyptr = keyptr[20:]
        dsa_p = keyptr[0:octets]
        keyptr = keyptr[octets:]
        dsa_g = keyptr[0:octets]
        keyptr = keyptr[octets:]
        dsa_y = keyptr[0:octets]
        try:
            dsa_public_key = dsa.DSAPublicNumbers(  # type: ignore
                _bytes_to_long(dsa_y),
                dsa.DSAParameterNumbers(
                    _bytes_to_long(dsa_p), _bytes_to_long(dsa_q), _bytes_to_long(dsa_g)
                ),
            ).public_key(default_backend())
        except ValueError:
            raise ValidationFailure("invalid public key")
        dsa_public_key.verify(sig, data, chosen_hash)
    elif _is_ecdsa(key.algorithm):
        keyptr = key.key
        curve: Any
        if key.algorithm == Algorithm.ECDSAP256SHA256:
            curve = ec.SECP256R1()
            octets = 32
        else:
            curve = ec.SECP384R1()
            octets = 48
        ecdsa_x = keyptr[0:octets]
        ecdsa_y = keyptr[octets : octets * 2]
        try:
            ecdsa_public_key = ec.EllipticCurvePublicNumbers(
                curve=curve, x=_bytes_to_long(ecdsa_x), y=_bytes_to_long(ecdsa_y)
            ).public_key(default_backend())
        except ValueError:
            raise ValidationFailure("invalid public key")
        ecdsa_public_key.verify(sig, data, ec.ECDSA(chosen_hash))
    elif _is_eddsa(key.algorithm):
        keyptr = key.key
        loader: Any
        if key.algorithm == Algorithm.ED25519:
            loader = ed25519.Ed25519PublicKey
        else:
            loader = ed448.Ed448PublicKey
        try:
            eddsa_public_key = loader.from_public_bytes(keyptr)
        except ValueError:
            raise ValidationFailure("invalid public key")
        eddsa_public_key.verify(sig, data)
    elif _is_gost(key.algorithm):
        raise UnsupportedAlgorithm(
            'algorithm "%s" not supported by dnspython'
            % algorithm_to_text(key.algorithm)
        )
    else:
        raise ValidationFailure("unknown algorithm %u" % key.algorithm)


def _validate_rrsig(
    rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
    rrsig: RRSIG,
    keys: Dict[dns.name.Name, Union[dns.node.Node, dns.rdataset.Rdataset]],
    origin: Optional[dns.name.Name] = None,
    now: Optional[float] = None,
    policy: Optional[Policy] = None,
) -> None:
    """Validate an RRset against a single signature rdata, throwing an
    exception if validation is not successful.

    *rrset*, the RRset to validate.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *rrsig*, a ``dns.rdata.Rdata``, the signature to validate.

    *keys*, the key dictionary, used to find the DNSKEY associated
    with a given name.  The dictionary is keyed by a
    ``dns.name.Name``, and has ``dns.node.Node`` or
    ``dns.rdataset.Rdataset`` values.

    *origin*, a ``dns.name.Name`` or ``None``, the origin to use for relative
    names.

    *now*, a ``float`` or ``None``, the time, in seconds since the epoch, to
    use as the current time when validating.  If ``None``, the actual current
    time is used.

    *policy*, a ``dns.dnssec.Policy`` or ``None``.  If ``None``, the default policy,
    ``dns.dnssec.default_policy`` is used; this policy defaults to that of RFC 8624.

    Raises ``ValidationFailure`` if the signature is expired, not yet valid,
    the public key is invalid, the algorithm is unknown, the verification
    fails, etc.

    Raises ``UnsupportedAlgorithm`` if the algorithm is recognized by
    dnspython but not implemented.
    """

    if policy is None:
        policy = default_policy

    candidate_keys = _find_candidate_keys(keys, rrsig)
    if candidate_keys is None:
        raise ValidationFailure("unknown key")

    if now is None:
        now = time.time()
    if rrsig.expiration < now:
        raise ValidationFailure("expired")
    if rrsig.inception > now:
        raise ValidationFailure("not yet valid")

    if _is_dsa(rrsig.algorithm):
        sig_r = rrsig.signature[1:21]
        sig_s = rrsig.signature[21:]
        sig = utils.encode_dss_signature(_bytes_to_long(sig_r), _bytes_to_long(sig_s))
    elif _is_ecdsa(rrsig.algorithm):
        if rrsig.algorithm == Algorithm.ECDSAP256SHA256:
            octets = 32
        else:
            octets = 48
        sig_r = rrsig.signature[0:octets]
        sig_s = rrsig.signature[octets:]
        sig = utils.encode_dss_signature(_bytes_to_long(sig_r), _bytes_to_long(sig_s))
    else:
        sig = rrsig.signature

    data = _make_rrsig_signature_data(rrset, rrsig, origin)
    chosen_hash = _make_hash(rrsig.algorithm)

    for candidate_key in candidate_keys:
        if not policy.ok_to_validate(candidate_key):
            continue
        try:
            _validate_signature(sig, data, candidate_key, chosen_hash)
            return
        except (InvalidSignature, ValidationFailure):
            # this happens on an individual validation failure
            continue
    # nothing verified -- raise failure:
    raise ValidationFailure("verify failure")


def _validate(
    rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
    rrsigset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
    keys: Dict[dns.name.Name, Union[dns.node.Node, dns.rdataset.Rdataset]],
    origin: Optional[dns.name.Name] = None,
    now: Optional[float] = None,
    policy: Optional[Policy] = None,
) -> None:
    """Validate an RRset against a signature RRset, throwing an exception
    if none of the signatures validate.

    *rrset*, the RRset to validate.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *rrsigset*, the signature RRset.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *keys*, the key dictionary, used to find the DNSKEY associated
    with a given name.  The dictionary is keyed by a
    ``dns.name.Name``, and has ``dns.node.Node`` or
    ``dns.rdataset.Rdataset`` values.

    *origin*, a ``dns.name.Name``, the origin to use for relative names;
    defaults to None.

    *now*, an ``int`` or ``None``, the time, in seconds since the epoch, to
    use as the current time when validating.  If ``None``, the actual current
    time is used.

    *policy*, a ``dns.dnssec.Policy`` or ``None``.  If ``None``, the default policy,
    ``dns.dnssec.default_policy`` is used; this policy defaults to that of RFC 8624.

    Raises ``ValidationFailure`` if the signature is expired, not yet valid,
    the public key is invalid, the algorithm is unknown, the verification
    fails, etc.
    """

    if policy is None:
        policy = default_policy

    if isinstance(origin, str):
        origin = dns.name.from_text(origin, dns.name.root)

    if isinstance(rrset, tuple):
        rrname = rrset[0]
    else:
        rrname = rrset.name

    if isinstance(rrsigset, tuple):
        rrsigname = rrsigset[0]
        rrsigrdataset = rrsigset[1]
    else:
        rrsigname = rrsigset.name
        rrsigrdataset = rrsigset

    rrname = rrname.choose_relativity(origin)
    rrsigname = rrsigname.choose_relativity(origin)
    if rrname != rrsigname:
        raise ValidationFailure("owner names do not match")

    for rrsig in rrsigrdataset:
        if not isinstance(rrsig, RRSIG):
            raise ValidationFailure("expected an RRSIG")
        try:
            _validate_rrsig(rrset, rrsig, keys, origin, now, policy)
            return
        except (ValidationFailure, UnsupportedAlgorithm):
            pass
    raise ValidationFailure("no RRSIGs validated")


def _sign(
    rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
    private_key: PrivateKey,
    signer: dns.name.Name,
    dnskey: DNSKEY,
    inception: Optional[Union[datetime, str, int, float]] = None,
    expiration: Optional[Union[datetime, str, int, float]] = None,
    lifetime: Optional[int] = None,
    verify: bool = False,
    policy: Optional[Policy] = None,
) -> RRSIG:
    """Sign RRset using private key.

    *rrset*, the RRset to validate.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *private_key*, the private key to use for signing, a
    ``cryptography.hazmat.primitives.asymmetric`` private key class applicable
    for DNSSEC.

    *signer*, a ``dns.name.Name``, the Signer's name.

    *dnskey*, a ``DNSKEY`` matching ``private_key``.

    *inception*, a ``datetime``, ``str``, ``int``, ``float`` or ``None``, the
    signature inception time.  If ``None``, the current time is used.  If a ``str``, the
    format is "YYYYMMDDHHMMSS" or alternatively the number of seconds since the UNIX
    epoch in text form; this is the same the RRSIG rdata's text form.
    Values of type `int` or `float` are interpreted as seconds since the UNIX epoch.

    *expiration*, a ``datetime``, ``str``, ``int``, ``float`` or ``None``, the signature
    expiration time.  If ``None``, the expiration time will be the inception time plus
    the value of the *lifetime* parameter.  See the description of *inception* above
    for how the various parameter types are interpreted.

    *lifetime*, an ``int`` or ``None``, the signature lifetime in seconds.  This
    parameter is only meaningful if *expiration* is ``None``.

    *verify*, a ``bool``.  If set to ``True``, the signer will verify signatures
    after they are created; the default is ``False``.

    *policy*, a ``dns.dnssec.Policy`` or ``None``.  If ``None``, the default policy,
    ``dns.dnssec.default_policy`` is used; this policy defaults to that of RFC 8624.

    Raises ``DeniedByPolicy`` if the signature is denied by policy.
    """

    if policy is None:
        policy = default_policy
    if not policy.ok_to_sign(dnskey):
        raise DeniedByPolicy

    if isinstance(rrset, tuple):
        rdclass = rrset[1].rdclass
        rdtype = rrset[1].rdtype
        rrname = rrset[0]
        original_ttl = rrset[1].ttl
    else:
        rdclass = rrset.rdclass
        rdtype = rrset.rdtype
        rrname = rrset.name
        original_ttl = rrset.ttl

    if inception is not None:
        rrsig_inception = to_timestamp(inception)
    else:
        rrsig_inception = int(time.time())

    if expiration is not None:
        rrsig_expiration = to_timestamp(expiration)
    elif lifetime is not None:
        rrsig_expiration = int(time.time()) + lifetime
    else:
        raise ValueError("expiration or lifetime must be specified")

    rrsig_template = RRSIG(
        rdclass=rdclass,
        rdtype=dns.rdatatype.RRSIG,
        type_covered=rdtype,
        algorithm=dnskey.algorithm,
        labels=len(rrname) - 1,
        original_ttl=original_ttl,
        expiration=rrsig_expiration,
        inception=rrsig_inception,
        key_tag=key_id(dnskey),
        signer=signer,
        signature=b"",
    )

    data = dns.dnssec._make_rrsig_signature_data(rrset, rrsig_template)
    chosen_hash = _make_hash(rrsig_template.algorithm)
    signature = None

    if isinstance(private_key, rsa.RSAPrivateKey):
        if not _is_rsa(dnskey.algorithm):
            raise ValueError("Invalid DNSKEY algorithm for RSA key")
        signature = private_key.sign(data, padding.PKCS1v15(), chosen_hash)
        if verify:
            private_key.public_key().verify(
                signature, data, padding.PKCS1v15(), chosen_hash
            )
    elif isinstance(private_key, dsa.DSAPrivateKey):
        if not _is_dsa(dnskey.algorithm):
            raise ValueError("Invalid DNSKEY algorithm for DSA key")
        public_dsa_key = private_key.public_key()
        if public_dsa_key.key_size > 1024:
            raise ValueError("DSA key size overflow")
        der_signature = private_key.sign(data, chosen_hash)
        if verify:
            public_dsa_key.verify(der_signature, data, chosen_hash)
        dsa_r, dsa_s = utils.decode_dss_signature(der_signature)
        dsa_t = (public_dsa_key.key_size // 8 - 64) // 8
        octets = 20
        signature = (
            struct.pack("!B", dsa_t)
            + int.to_bytes(dsa_r, length=octets, byteorder="big")
            + int.to_bytes(dsa_s, length=octets, byteorder="big")
        )
    elif isinstance(private_key, ec.EllipticCurvePrivateKey):
        if not _is_ecdsa(dnskey.algorithm):
            raise ValueError("Invalid DNSKEY algorithm for EC key")
        der_signature = private_key.sign(data, ec.ECDSA(chosen_hash))
        if verify:
            private_key.public_key().verify(der_signature, data, ec.ECDSA(chosen_hash))
        if dnskey.algorithm == Algorithm.ECDSAP256SHA256:
            octets = 32
        else:
            octets = 48
        dsa_r, dsa_s = utils.decode_dss_signature(der_signature)
        signature = int.to_bytes(dsa_r, length=octets, byteorder="big") + int.to_bytes(
            dsa_s, length=octets, byteorder="big"
        )
    elif isinstance(private_key, ed25519.Ed25519PrivateKey):
        if dnskey.algorithm != Algorithm.ED25519:
            raise ValueError("Invalid DNSKEY algorithm for ED25519 key")
        signature = private_key.sign(data)
        if verify:
            private_key.public_key().verify(signature, data)
    elif isinstance(private_key, ed448.Ed448PrivateKey):
        if dnskey.algorithm != Algorithm.ED448:
            raise ValueError("Invalid DNSKEY algorithm for ED448 key")
        signature = private_key.sign(data)
        if verify:
            private_key.public_key().verify(signature, data)
    else:
        raise TypeError("Unsupported key algorithm")

    return cast(RRSIG, rrsig_template.replace(signature=signature))


def _make_rrsig_signature_data(
    rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
    rrsig: RRSIG,
    origin: Optional[dns.name.Name] = None,
) -> bytes:
    """Create signature rdata.

    *rrset*, the RRset to sign/validate.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *rrsig*, a ``dns.rdata.Rdata``, the signature to validate, or the
    signature template used when signing.

    *origin*, a ``dns.name.Name`` or ``None``, the origin to use for relative
    names.

    Raises ``UnsupportedAlgorithm`` if the algorithm is recognized by
    dnspython but not implemented.
    """

    if isinstance(origin, str):
        origin = dns.name.from_text(origin, dns.name.root)

    signer = rrsig.signer
    if not signer.is_absolute():
        if origin is None:
            raise ValidationFailure("relative RR name without an origin specified")
        signer = signer.derelativize(origin)

    # For convenience, allow the rrset to be specified as a (name,
    # rdataset) tuple as well as a proper rrset
    rrname, rdataset = _get_rrname_rdataset(rrset)

    data = b""
    data += rrsig.to_wire(origin=signer)[:18]
    data += rrsig.signer.to_digestable(signer)

    # Derelativize the name before considering labels.
    if not rrname.is_absolute():
        if origin is None:
            raise ValidationFailure("relative RR name without an origin specified")
        rrname = rrname.derelativize(origin)

    if len(rrname) - 1 < rrsig.labels:
        raise ValidationFailure("owner name longer than RRSIG labels")
    elif rrsig.labels < len(rrname) - 1:
        suffix = rrname.split(rrsig.labels + 1)[1]
        rrname = dns.name.from_text("*", suffix)
    rrnamebuf = rrname.to_digestable()
    rrfixed = struct.pack("!HHI", rdataset.rdtype, rdataset.rdclass, rrsig.original_ttl)
    rdatas = [rdata.to_digestable(origin) for rdata in rdataset]
    for rdata in sorted(rdatas):
        data += rrnamebuf
        data += rrfixed
        rrlen = struct.pack("!H", len(rdata))
        data += rrlen
        data += rdata

    return data


def _make_dnskey(
    public_key: PublicKey,
    algorithm: Union[int, str],
    flags: int = Flag.ZONE,
    protocol: int = 3,
) -> DNSKEY:
    """Convert a public key to DNSKEY Rdata

    *public_key*, the public key to convert, a
    ``cryptography.hazmat.primitives.asymmetric`` public key class applicable
    for DNSSEC.

    *algorithm*, a ``str`` or ``int`` specifying the DNSKEY algorithm.

    *flags*: DNSKEY flags field as an integer.

    *protocol*: DNSKEY protocol field as an integer.

    Raises ``ValueError`` if the specified key algorithm parameters are not
    unsupported, ``TypeError`` if the key type is unsupported,
    `UnsupportedAlgorithm` if the algorithm is unknown and
    `AlgorithmKeyMismatch` if the algorithm does not match the key type.

    Return DNSKEY ``Rdata``.
    """

    def encode_rsa_public_key(public_key: "rsa.RSAPublicKey") -> bytes:
        """Encode a public key per RFC 3110, section 2."""
        pn = public_key.public_numbers()
        _exp_len = math.ceil(int.bit_length(pn.e) / 8)
        exp = int.to_bytes(pn.e, length=_exp_len, byteorder="big")
        if _exp_len > 255:
            exp_header = b"\0" + struct.pack("!H", _exp_len)
        else:
            exp_header = struct.pack("!B", _exp_len)
        if pn.n.bit_length() < 512 or pn.n.bit_length() > 4096:
            raise ValueError("unsupported RSA key length")
        return exp_header + exp + pn.n.to_bytes((pn.n.bit_length() + 7) // 8, "big")

    def encode_dsa_public_key(public_key: "dsa.DSAPublicKey") -> bytes:
        """Encode a public key per RFC 2536, section 2."""
        pn = public_key.public_numbers()
        dsa_t = (public_key.key_size // 8 - 64) // 8
        if dsa_t > 8:
            raise ValueError("unsupported DSA key size")
        octets = 64 + dsa_t * 8
        res = struct.pack("!B", dsa_t)
        res += pn.parameter_numbers.q.to_bytes(20, "big")
        res += pn.parameter_numbers.p.to_bytes(octets, "big")
        res += pn.parameter_numbers.g.to_bytes(octets, "big")
        res += pn.y.to_bytes(octets, "big")
        return res

    def encode_ecdsa_public_key(public_key: "ec.EllipticCurvePublicKey") -> bytes:
        """Encode a public key per RFC 6605, section 4."""
        pn = public_key.public_numbers()
        if isinstance(public_key.curve, ec.SECP256R1):
            return pn.x.to_bytes(32, "big") + pn.y.to_bytes(32, "big")
        elif isinstance(public_key.curve, ec.SECP384R1):
            return pn.x.to_bytes(48, "big") + pn.y.to_bytes(48, "big")
        else:
            raise ValueError("unsupported ECDSA curve")

    the_algorithm = Algorithm.make(algorithm)

    _ensure_algorithm_key_combination(the_algorithm, public_key)

    if isinstance(public_key, rsa.RSAPublicKey):
        key_bytes = encode_rsa_public_key(public_key)
    elif isinstance(public_key, dsa.DSAPublicKey):
        key_bytes = encode_dsa_public_key(public_key)
    elif isinstance(public_key, ec.EllipticCurvePublicKey):
        key_bytes = encode_ecdsa_public_key(public_key)
    elif isinstance(public_key, ed25519.Ed25519PublicKey):
        key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    elif isinstance(public_key, ed448.Ed448PublicKey):
        key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
    else:
        raise TypeError("unsupported key algorithm")

    return DNSKEY(
        rdclass=dns.rdataclass.IN,
        rdtype=dns.rdatatype.DNSKEY,
        flags=flags,
        protocol=protocol,
        algorithm=the_algorithm,
        key=key_bytes,
    )


def _make_cdnskey(
    public_key: PublicKey,
    algorithm: Union[int, str],
    flags: int = Flag.ZONE,
    protocol: int = 3,
) -> CDNSKEY:
    """Convert a public key to CDNSKEY Rdata

    *public_key*, the public key to convert, a
    ``cryptography.hazmat.primitives.asymmetric`` public key class applicable
    for DNSSEC.

    *algorithm*, a ``str`` or ``int`` specifying the DNSKEY algorithm.

    *flags*: DNSKEY flags field as an integer.

    *protocol*: DNSKEY protocol field as an integer.

    Raises ``ValueError`` if the specified key algorithm parameters are not
    unsupported, ``TypeError`` if the key type is unsupported,
    `UnsupportedAlgorithm` if the algorithm is unknown and
    `AlgorithmKeyMismatch` if the algorithm does not match the key type.

    Return CDNSKEY ``Rdata``.
    """

    dnskey = _make_dnskey(public_key, algorithm, flags, protocol)

    return CDNSKEY(
        rdclass=dnskey.rdclass,
        rdtype=dns.rdatatype.CDNSKEY,
        flags=dnskey.flags,
        protocol=dnskey.protocol,
        algorithm=dnskey.algorithm,
        key=dnskey.key,
    )


def nsec3_hash(
    domain: Union[dns.name.Name, str],
    salt: Optional[Union[str, bytes]],
    iterations: int,
    algorithm: Union[int, str],
) -> str:
    """
    Calculate the NSEC3 hash, according to
    https://tools.ietf.org/html/rfc5155#section-5

    *domain*, a ``dns.name.Name`` or ``str``, the name to hash.

    *salt*, a ``str``, ``bytes``, or ``None``, the hash salt.  If a
    string, it is decoded as a hex string.

    *iterations*, an ``int``, the number of iterations.

    *algorithm*, a ``str`` or ``int``, the hash algorithm.
    The only defined algorithm is SHA1.

    Returns a ``str``, the encoded NSEC3 hash.
    """

    b32_conversion = str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", "0123456789ABCDEFGHIJKLMNOPQRSTUV"
    )

    try:
        if isinstance(algorithm, str):
            algorithm = NSEC3Hash[algorithm.upper()]
    except Exception:
        raise ValueError("Wrong hash algorithm (only SHA1 is supported)")

    if algorithm != NSEC3Hash.SHA1:
        raise ValueError("Wrong hash algorithm (only SHA1 is supported)")

    if salt is None:
        salt_encoded = b""
    elif isinstance(salt, str):
        if len(salt) % 2 == 0:
            salt_encoded = bytes.fromhex(salt)
        else:
            raise ValueError("Invalid salt length")
    else:
        salt_encoded = salt

    if not isinstance(domain, dns.name.Name):
        domain = dns.name.from_text(domain)
    domain_encoded = domain.canonicalize().to_wire()
    assert domain_encoded is not None

    digest = hashlib.sha1(domain_encoded + salt_encoded).digest()
    for _ in range(iterations):
        digest = hashlib.sha1(digest + salt_encoded).digest()

    output = base64.b32encode(digest).decode("utf-8")
    output = output.translate(b32_conversion)

    return output


def make_ds_rdataset(
    rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]],
    algorithms: Set[Union[DSDigest, str]],
    origin: Optional[dns.name.Name] = None,
) -> dns.rdataset.Rdataset:
    """Create a DS record from DNSKEY/CDNSKEY/CDS.

    *rrset*, the RRset to create DS Rdataset for.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *algorithms*, a set of ``str`` or ``int`` specifying the hash algorithms.
    The currently supported hashes are "SHA1", "SHA256", and "SHA384". Case
    does not matter for these strings. If the RRset is a CDS, only digest
    algorithms matching algorithms are accepted.

    *origin*, a ``dns.name.Name`` or ``None``.  If `key` is a relative name,
    then it will be made absolute using the specified origin.

    Raises ``UnsupportedAlgorithm`` if any of the algorithms are unknown and
    ``ValueError`` if the given RRset is not usable.

    Returns a ``dns.rdataset.Rdataset``
    """

    rrname, rdataset = _get_rrname_rdataset(rrset)

    if rdataset.rdtype not in (
        dns.rdatatype.DNSKEY,
        dns.rdatatype.CDNSKEY,
        dns.rdatatype.CDS,
    ):
        raise ValueError("rrset not a DNSKEY/CDNSKEY/CDS")

    _algorithms = set()
    for algorithm in algorithms:
        try:
            if isinstance(algorithm, str):
                algorithm = DSDigest[algorithm.upper()]
        except Exception:
            raise UnsupportedAlgorithm('unsupported algorithm "%s"' % algorithm)
        _algorithms.add(algorithm)

    if rdataset.rdtype == dns.rdatatype.CDS:
        res = []
        for rdata in cds_rdataset_to_ds_rdataset(rdataset):
            if rdata.digest_type in _algorithms:
                res.append(rdata)
        if len(res) == 0:
            raise ValueError("no acceptable CDS rdata found")
        return dns.rdataset.from_rdata_list(rdataset.ttl, res)

    res = []
    for algorithm in _algorithms:
        res.extend(dnskey_rdataset_to_cds_rdataset(rrname, rdataset, algorithm, origin))
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)


def cds_rdataset_to_ds_rdataset(
    rdataset: dns.rdataset.Rdataset,
) -> dns.rdataset.Rdataset:
    """Create a CDS record from DS.

    *rdataset*, a ``dns.rdataset.Rdataset``, to create DS Rdataset for.

    Raises ``ValueError`` if the rdataset is not CDS.

    Returns a ``dns.rdataset.Rdataset``
    """

    if rdataset.rdtype != dns.rdatatype.CDS:
        raise ValueError("rdataset not a CDS")
    res = []
    for rdata in rdataset:
        res.append(
            CDS(
                rdclass=rdata.rdclass,
                rdtype=dns.rdatatype.DS,
                key_tag=rdata.key_tag,
                algorithm=rdata.algorithm,
                digest_type=rdata.digest_type,
                digest=rdata.digest,
            )
        )
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)


def dnskey_rdataset_to_cds_rdataset(
    name: Union[dns.name.Name, str],
    rdataset: dns.rdataset.Rdataset,
    algorithm: Union[DSDigest, str],
    origin: Optional[dns.name.Name] = None,
) -> dns.rdataset.Rdataset:
    """Create a CDS record from DNSKEY/CDNSKEY.

    *name*, a ``dns.name.Name`` or ``str``, the owner name of the CDS record.

    *rdataset*, a ``dns.rdataset.Rdataset``, to create DS Rdataset for.

    *algorithm*, a ``str`` or ``int`` specifying the hash algorithm.
    The currently supported hashes are "SHA1", "SHA256", and "SHA384". Case
    does not matter for these strings.

    *origin*, a ``dns.name.Name`` or ``None``.  If `key` is a relative name,
    then it will be made absolute using the specified origin.

    Raises ``UnsupportedAlgorithm`` if the algorithm is unknown or
    ``ValueError`` if the rdataset is not DNSKEY/CDNSKEY.

    Returns a ``dns.rdataset.Rdataset``
    """

    if rdataset.rdtype not in (dns.rdatatype.DNSKEY, dns.rdatatype.CDNSKEY):
        raise ValueError("rdataset not a DNSKEY/CDNSKEY")
    res = []
    for rdata in rdataset:
        res.append(make_cds(name, rdata, algorithm, origin))
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)


def dnskey_rdataset_to_cdnskey_rdataset(
    rdataset: dns.rdataset.Rdataset,
) -> dns.rdataset.Rdataset:
    """Create a CDNSKEY record from DNSKEY.

    *rdataset*, a ``dns.rdataset.Rdataset``, to create CDNSKEY Rdataset for.

    Returns a ``dns.rdataset.Rdataset``
    """

    if rdataset.rdtype != dns.rdatatype.DNSKEY:
        raise ValueError("rdataset not a DNSKEY")
    res = []
    for rdata in rdataset:
        res.append(
            CDNSKEY(
                rdclass=rdataset.rdclass,
                rdtype=rdataset.rdtype,
                flags=rdata.flags,
                protocol=rdata.protocol,
                algorithm=rdata.algorithm,
                key=rdata.key,
            )
        )
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)


def _need_pyca(*args, **kwargs):
    raise ImportError(
        "DNSSEC validation requires " + "python cryptography"
    )  # pragma: no cover


try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric import utils
    from cryptography.hazmat.primitives.asymmetric import dsa
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.asymmetric import ed448
    from cryptography.hazmat.primitives.asymmetric import rsa
except ImportError:  # pragma: no cover
    validate = _need_pyca
    validate_rrsig = _need_pyca
    sign = _need_pyca
    make_dnskey = _need_pyca
    make_cdnskey = _need_pyca
    _have_pyca = False
else:
    validate = _validate  # type: ignore
    validate_rrsig = _validate_rrsig  # type: ignore
    sign = _sign
    make_dnskey = _make_dnskey
    make_cdnskey = _make_cdnskey
    _have_pyca = True

### BEGIN generated Algorithm constants

RSAMD5 = Algorithm.RSAMD5
DH = Algorithm.DH
DSA = Algorithm.DSA
ECC = Algorithm.ECC
RSASHA1 = Algorithm.RSASHA1
DSANSEC3SHA1 = Algorithm.DSANSEC3SHA1
RSASHA1NSEC3SHA1 = Algorithm.RSASHA1NSEC3SHA1
RSASHA256 = Algorithm.RSASHA256
RSASHA512 = Algorithm.RSASHA512
ECCGOST = Algorithm.ECCGOST
ECDSAP256SHA256 = Algorithm.ECDSAP256SHA256
ECDSAP384SHA384 = Algorithm.ECDSAP384SHA384
ED25519 = Algorithm.ED25519
ED448 = Algorithm.ED448
INDIRECT = Algorithm.INDIRECT
PRIVATEDNS = Algorithm.PRIVATEDNS
PRIVATEOID = Algorithm.PRIVATEOID

### END generated Algorithm constants
