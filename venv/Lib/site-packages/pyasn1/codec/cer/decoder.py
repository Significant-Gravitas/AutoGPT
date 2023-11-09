#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from pyasn1 import error
from pyasn1.codec.ber import decoder
from pyasn1.compat.octets import oct2int
from pyasn1.type import univ

__all__ = ['decode']


class BooleanDecoder(decoder.AbstractSimpleDecoder):
    protoComponent = univ.Boolean(0)

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        head, tail = substrate[:length], substrate[length:]
        if not head or length != 1:
            raise error.PyAsn1Error('Not single-octet Boolean payload')
        byte = oct2int(head[0])
        # CER/DER specifies encoding of TRUE as 0xFF and FALSE as 0x0, while
        # BER allows any non-zero value as TRUE; cf. sections 8.2.2. and 11.1 
        # in https://www.itu.int/ITU-T/studygroups/com17/languages/X.690-0207.pdf
        if byte == 0xff:
            value = 1
        elif byte == 0x00:
            value = 0
        else:
            raise error.PyAsn1Error('Unexpected Boolean payload: %s' % byte)
        return self._createComponent(asn1Spec, tagSet, value, **options), tail

# TODO: prohibit non-canonical encoding
BitStringDecoder = decoder.BitStringDecoder
OctetStringDecoder = decoder.OctetStringDecoder
RealDecoder = decoder.RealDecoder

tagMap = decoder.tagMap.copy()
tagMap.update(
    {univ.Boolean.tagSet: BooleanDecoder(),
     univ.BitString.tagSet: BitStringDecoder(),
     univ.OctetString.tagSet: OctetStringDecoder(),
     univ.Real.tagSet: RealDecoder()}
)

typeMap = decoder.typeMap.copy()

# Put in non-ambiguous types for faster codec lookup
for typeDecoder in tagMap.values():
    if typeDecoder.protoComponent is not None:
        typeId = typeDecoder.protoComponent.__class__.typeId
        if typeId is not None and typeId not in typeMap:
            typeMap[typeId] = typeDecoder


class Decoder(decoder.Decoder):
    pass


#: Turns CER octet stream into an ASN.1 object.
#:
#: Takes CER octet-stream and decode it into an ASN.1 object
#: (e.g. :py:class:`~pyasn1.type.base.PyAsn1Item` derivative) which
#: may be a scalar or an arbitrary nested structure.
#:
#: Parameters
#: ----------
#: substrate: :py:class:`bytes` (Python 3) or :py:class:`str` (Python 2)
#:     CER octet-stream
#:
#: Keyword Args
#: ------------
#: asn1Spec: any pyasn1 type object e.g. :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
#:     A pyasn1 type object to act as a template guiding the decoder. Depending on the ASN.1 structure
#:     being decoded, *asn1Spec* may or may not be required. Most common reason for
#:     it to require is that ASN.1 structure is encoded in *IMPLICIT* tagging mode.
#:
#: Returns
#: -------
#: : :py:class:`tuple`
#:     A tuple of pyasn1 object recovered from CER substrate (:py:class:`~pyasn1.type.base.PyAsn1Item` derivative)
#:     and the unprocessed trailing portion of the *substrate* (may be empty)
#:
#: Raises
#: ------
#: ~pyasn1.error.PyAsn1Error, ~pyasn1.error.SubstrateUnderrunError
#:     On decoding errors
#:
#: Examples
#: --------
#: Decode CER serialisation without ASN.1 schema
#:
#: .. code-block:: pycon
#:
#:    >>> s, _ = decode(b'0\x80\x02\x01\x01\x02\x01\x02\x02\x01\x03\x00\x00')
#:    >>> str(s)
#:    SequenceOf:
#:     1 2 3
#:
#: Decode CER serialisation with ASN.1 schema
#:
#: .. code-block:: pycon
#:
#:    >>> seq = SequenceOf(componentType=Integer())
#:    >>> s, _ = decode(b'0\x80\x02\x01\x01\x02\x01\x02\x02\x01\x03\x00\x00', asn1Spec=seq)
#:    >>> str(s)
#:    SequenceOf:
#:     1 2 3
#:
decode = Decoder(tagMap, decoder.typeMap)
