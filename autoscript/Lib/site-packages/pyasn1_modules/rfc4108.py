#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add items from the verified errata.
# Modified by Russ Housley to add maps for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# CMS Firmware Wrapper
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc4108.txt
# https://www.rfc-editor.org/errata_search.php?rfc=4108
#


from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652

MAX = float('inf')


class HardwareSerialEntry(univ.Choice):
    pass

HardwareSerialEntry.componentType = namedtype.NamedTypes(
    namedtype.NamedType('all', univ.Null()),
    namedtype.NamedType('single', univ.OctetString()),
    namedtype.NamedType('block', univ.Sequence(componentType=namedtype.NamedTypes(
        namedtype.NamedType('low', univ.OctetString()),
        namedtype.NamedType('high', univ.OctetString())
    ))
    )
)


class HardwareModules(univ.Sequence):
    pass

HardwareModules.componentType = namedtype.NamedTypes(
    namedtype.NamedType('hwType', univ.ObjectIdentifier()),
    namedtype.NamedType('hwSerialEntries', univ.SequenceOf(componentType=HardwareSerialEntry()))
)


class CommunityIdentifier(univ.Choice):
    pass

CommunityIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('communityOID', univ.ObjectIdentifier()),
    namedtype.NamedType('hwModuleList', HardwareModules())
)



class PreferredPackageIdentifier(univ.Sequence):
    pass

PreferredPackageIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('fwPkgID', univ.ObjectIdentifier()),
    namedtype.NamedType('verNum', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, MAX)))
)


class PreferredOrLegacyPackageIdentifier(univ.Choice):
    pass

PreferredOrLegacyPackageIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('preferred', PreferredPackageIdentifier()),
    namedtype.NamedType('legacy', univ.OctetString())
)


class CurrentFWConfig(univ.Sequence):
    pass

CurrentFWConfig.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('fwPkgType', univ.Integer()),
    namedtype.NamedType('fwPkgName', PreferredOrLegacyPackageIdentifier())
)


class PreferredOrLegacyStalePackageIdentifier(univ.Choice):
    pass

PreferredOrLegacyStalePackageIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('preferredStaleVerNum', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, MAX))),
    namedtype.NamedType('legacyStaleVersion', univ.OctetString())
)


class FirmwarePackageLoadErrorCode(univ.Enumerated):
    pass

FirmwarePackageLoadErrorCode.namedValues = namedval.NamedValues(
    ('decodeFailure', 1),
    ('badContentInfo', 2),
    ('badSignedData', 3),
    ('badEncapContent', 4),
    ('badCertificate', 5),
    ('badSignerInfo', 6),
    ('badSignedAttrs', 7),
    ('badUnsignedAttrs', 8),
    ('missingContent', 9),
    ('noTrustAnchor', 10),
    ('notAuthorized', 11),
    ('badDigestAlgorithm', 12),
    ('badSignatureAlgorithm', 13),
    ('unsupportedKeySize', 14),
    ('signatureFailure', 15),
    ('contentTypeMismatch', 16),
    ('badEncryptedData', 17),
    ('unprotectedAttrsPresent', 18),
    ('badEncryptContent', 19),
    ('badEncryptAlgorithm', 20),
    ('missingCiphertext', 21),
    ('noDecryptKey', 22),
    ('decryptFailure', 23),
    ('badCompressAlgorithm', 24),
    ('missingCompressedContent', 25),
    ('decompressFailure', 26),
    ('wrongHardware', 27),
    ('stalePackage', 28),
    ('notInCommunity', 29),
    ('unsupportedPackageType', 30),
    ('missingDependency', 31),
    ('wrongDependencyVersion', 32),
    ('insufficientMemory', 33),
    ('badFirmware', 34),
    ('unsupportedParameters', 35),
    ('breaksDependency', 36),
    ('otherError', 99)
)


class VendorLoadErrorCode(univ.Integer):
    pass


# Wrapped Firmware Key Unsigned Attribute and Object Identifier

id_aa_wrappedFirmwareKey = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.39')

class WrappedFirmwareKey(rfc5652.EnvelopedData):
    pass


# Firmware Package Information Signed Attribute and Object Identifier

id_aa_firmwarePackageInfo = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.42')

class FirmwarePackageInfo(univ.Sequence):
    pass

FirmwarePackageInfo.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('fwPkgType', univ.Integer()),
    namedtype.OptionalNamedType('dependencies', univ.SequenceOf(componentType=PreferredOrLegacyPackageIdentifier()))
)

FirmwarePackageInfo.sizeSpec = univ.Sequence.sizeSpec + constraint.ValueSizeConstraint(1, 2)


# Community Identifiers Signed Attribute and Object Identifier

id_aa_communityIdentifiers = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.40')

class CommunityIdentifiers(univ.SequenceOf):
    pass

CommunityIdentifiers.componentType = CommunityIdentifier()


# Implemented Compression Algorithms Signed Attribute and Object Identifier

id_aa_implCompressAlgs = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.43')

class ImplementedCompressAlgorithms(univ.SequenceOf):
    pass

ImplementedCompressAlgorithms.componentType = univ.ObjectIdentifier()


# Implemented Cryptographic Algorithms Signed Attribute and Object Identifier

id_aa_implCryptoAlgs = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.38')

class ImplementedCryptoAlgorithms(univ.SequenceOf):
    pass

ImplementedCryptoAlgorithms.componentType = univ.ObjectIdentifier()


# Decrypt Key Identifier Signed Attribute and Object Identifier

id_aa_decryptKeyID = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.37')

class DecryptKeyIdentifier(univ.OctetString):
    pass


# Target Hardware Identifier Signed Attribute and Object Identifier

id_aa_targetHardwareIDs = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.36')

class TargetHardwareIdentifiers(univ.SequenceOf):
    pass

TargetHardwareIdentifiers.componentType = univ.ObjectIdentifier()


# Firmware Package Identifier Signed Attribute and Object Identifier

id_aa_firmwarePackageID = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.35')

class FirmwarePackageIdentifier(univ.Sequence):
    pass

FirmwarePackageIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('name', PreferredOrLegacyPackageIdentifier()),
    namedtype.OptionalNamedType('stale', PreferredOrLegacyStalePackageIdentifier())
)


# Firmware Package Message Digest Signed Attribute and Object Identifier

id_aa_fwPkgMessageDigest = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.41')

class FirmwarePackageMessageDigest(univ.Sequence):
    pass

FirmwarePackageMessageDigest.componentType = namedtype.NamedTypes(
    namedtype.NamedType('algorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('msgDigest', univ.OctetString())
)


# Firmware Package Load Error Report Content Type and Object Identifier

class FWErrorVersion(univ.Integer):
    pass

FWErrorVersion.namedValues = namedval.NamedValues(
    ('v1', 1)
)


id_ct_firmwareLoadError = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.18')

class FirmwarePackageLoadError(univ.Sequence):
    pass

FirmwarePackageLoadError.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', FWErrorVersion().subtype(value='v1')),
    namedtype.NamedType('hwType', univ.ObjectIdentifier()),
    namedtype.NamedType('hwSerialNum', univ.OctetString()),
    namedtype.NamedType('errorCode', FirmwarePackageLoadErrorCode()),
    namedtype.OptionalNamedType('vendorErrorCode', VendorLoadErrorCode()),
    namedtype.OptionalNamedType('fwPkgName', PreferredOrLegacyPackageIdentifier()),
    namedtype.OptionalNamedType('config', univ.SequenceOf(componentType=CurrentFWConfig()).subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


# Firmware Package Load Receipt Content Type and Object Identifier

class FWReceiptVersion(univ.Integer):
    pass

FWReceiptVersion.namedValues = namedval.NamedValues(
    ('v1', 1)
)


id_ct_firmwareLoadReceipt = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.17')

class FirmwarePackageLoadReceipt(univ.Sequence):
    pass

FirmwarePackageLoadReceipt.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', FWReceiptVersion().subtype(value='v1')),
    namedtype.NamedType('hwType', univ.ObjectIdentifier()),
    namedtype.NamedType('hwSerialNum', univ.OctetString()),
    namedtype.NamedType('fwPkgName', PreferredOrLegacyPackageIdentifier()),
    namedtype.OptionalNamedType('trustAnchorKeyID', univ.OctetString()),
    namedtype.OptionalNamedType('decryptKeyID', univ.OctetString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


# Firmware Package Content Type and Object Identifier

id_ct_firmwarePackage = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.16')

class FirmwarePkgData(univ.OctetString):
    pass


# Other Name syntax for Hardware Module Name

id_on_hardwareModuleName = univ.ObjectIdentifier('1.3.6.1.5.5.7.8.4')

class HardwareModuleName(univ.Sequence):
    pass

HardwareModuleName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('hwType', univ.ObjectIdentifier()),
    namedtype.NamedType('hwSerialNum', univ.OctetString())
)


# Map of Attribute Type OIDs to Attributes is added to the
# ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_aa_wrappedFirmwareKey: WrappedFirmwareKey(),
    id_aa_firmwarePackageInfo: FirmwarePackageInfo(),
    id_aa_communityIdentifiers: CommunityIdentifiers(),
    id_aa_implCompressAlgs: ImplementedCompressAlgorithms(),
    id_aa_implCryptoAlgs: ImplementedCryptoAlgorithms(),
    id_aa_decryptKeyID: DecryptKeyIdentifier(),
    id_aa_targetHardwareIDs: TargetHardwareIdentifiers(),
    id_aa_firmwarePackageID: FirmwarePackageIdentifier(),
    id_aa_fwPkgMessageDigest: FirmwarePackageMessageDigest(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)


# Map of Content Type OIDs to Content Types is added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_firmwareLoadError: FirmwarePackageLoadError(),
    id_ct_firmwareLoadReceipt: FirmwarePackageLoadReceipt(),
    id_ct_firmwarePackage: FirmwarePkgData(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)


# Map of Other Name OIDs to Other Name is added to the
# ones that are in rfc5280.py

_anotherNameMapUpdate = {
    id_on_hardwareModuleName: HardwareModuleName(),
}

rfc5280.anotherNameMap.update(_anotherNameMapUpdate)
