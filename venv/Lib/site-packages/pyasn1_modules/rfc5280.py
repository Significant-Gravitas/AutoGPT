# coding: utf-8
#
# This file is part of pyasn1-modules software.
#
# Created by Stanisław Pitucha with asn1ate tool.
# Updated by Russ Housley for ORAddress Extension Attribute opentype support.
# Updated by Russ Housley for AlgorithmIdentifier opentype support.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# Internet X.509 Public Key Infrastructure Certificate and Certificate
# Revocation List (CRL) Profile
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5280.txt
#
from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

MAX = float('inf')


def _buildOid(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


ub_e163_4_sub_address_length = univ.Integer(40)

ub_e163_4_number_length = univ.Integer(15)

unformatted_postal_address = univ.Integer(16)


class TerminalType(univ.Integer):
    pass


TerminalType.namedValues = namedval.NamedValues(
    ('telex', 3),
    ('teletex', 4),
    ('g3-facsimile', 5),
    ('g4-facsimile', 6),
    ('ia5-terminal', 7),
    ('videotex', 8)
)


class Extension(univ.Sequence):
    pass


Extension.componentType = namedtype.NamedTypes(
    namedtype.NamedType('extnID', univ.ObjectIdentifier()),
    namedtype.DefaultedNamedType('critical', univ.Boolean().subtype(value=0)),
    namedtype.NamedType('extnValue', univ.OctetString())
)


class Extensions(univ.SequenceOf):
    pass


Extensions.componentType = Extension()
Extensions.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

physical_delivery_personal_name = univ.Integer(13)

ub_unformatted_address_length = univ.Integer(180)

ub_pds_parameter_length = univ.Integer(30)

ub_pds_physical_address_lines = univ.Integer(6)


class UnformattedPostalAddress(univ.Set):
    pass


UnformattedPostalAddress.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('printable-address', univ.SequenceOf(componentType=char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_parameter_length)))),
    namedtype.OptionalNamedType('teletex-string', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_unformatted_address_length)))
)

ub_organization_name = univ.Integer(64)


class X520OrganizationName(univ.Choice):
    pass


X520OrganizationName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
    namedtype.NamedType('printableString', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
    namedtype.NamedType('universalString', char.UniversalString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
    namedtype.NamedType('bmpString',
                        char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name)))
)

ub_x121_address_length = univ.Integer(16)

pds_name = univ.Integer(7)

id_pkix = _buildOid(1, 3, 6, 1, 5, 5, 7)

id_kp = _buildOid(id_pkix, 3)

ub_postal_code_length = univ.Integer(16)


class PostalCode(univ.Choice):
    pass


PostalCode.componentType = namedtype.NamedTypes(
    namedtype.NamedType('numeric-code', char.NumericString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_postal_code_length))),
    namedtype.NamedType('printable-code', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_postal_code_length)))
)

ub_generation_qualifier_length = univ.Integer(3)

unique_postal_name = univ.Integer(20)


class DomainComponent(char.IA5String):
    pass


ub_domain_defined_attribute_value_length = univ.Integer(128)

ub_match = univ.Integer(128)

id_at = _buildOid(2, 5, 4)


class AttributeType(univ.ObjectIdentifier):
    pass


id_at_organizationalUnitName = _buildOid(id_at, 11)

terminal_type = univ.Integer(23)


class PDSParameter(univ.Set):
    pass


PDSParameter.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('printable-string', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_parameter_length))),
    namedtype.OptionalNamedType('teletex-string', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_parameter_length)))
)


class PhysicalDeliveryPersonalName(PDSParameter):
    pass


ub_surname_length = univ.Integer(40)

id_ad = _buildOid(id_pkix, 48)

ub_domain_defined_attribute_type_length = univ.Integer(8)


class TeletexDomainDefinedAttribute(univ.Sequence):
    pass


TeletexDomainDefinedAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('type', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_type_length))),
    namedtype.NamedType('value', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_value_length)))
)

ub_domain_defined_attributes = univ.Integer(4)


class TeletexDomainDefinedAttributes(univ.SequenceOf):
    pass


TeletexDomainDefinedAttributes.componentType = TeletexDomainDefinedAttribute()
TeletexDomainDefinedAttributes.sizeSpec = constraint.ValueSizeConstraint(1, ub_domain_defined_attributes)

extended_network_address = univ.Integer(22)

ub_locality_name = univ.Integer(128)


class X520LocalityName(univ.Choice):
    pass


X520LocalityName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
    namedtype.NamedType('printableString', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
    namedtype.NamedType('universalString', char.UniversalString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
    namedtype.NamedType('bmpString',
                        char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name)))
)

teletex_organization_name = univ.Integer(3)

ub_given_name_length = univ.Integer(16)

ub_initials_length = univ.Integer(5)


class PersonalName(univ.Set):
    pass


PersonalName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('surname', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_surname_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('given-name', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_given_name_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('initials', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_initials_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('generation-qualifier', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_generation_qualifier_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)

ub_organizational_unit_name_length = univ.Integer(32)


class OrganizationalUnitName(char.PrintableString):
    pass


OrganizationalUnitName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_organizational_unit_name_length)

id_at_generationQualifier = _buildOid(id_at, 44)


class Version(univ.Integer):
    pass


Version.namedValues = namedval.NamedValues(
    ('v1', 0),
    ('v2', 1),
    ('v3', 2)
)


class CertificateSerialNumber(univ.Integer):
    pass


algorithmIdentifierMap = {}


class AlgorithmIdentifier(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('algorithm', univ.ObjectIdentifier()),
        namedtype.OptionalNamedType('parameters', univ.Any(),
            openType=opentype.OpenType('algorithm', algorithmIdentifierMap)
        )
    )


class Time(univ.Choice):
    pass


Time.componentType = namedtype.NamedTypes(
    namedtype.NamedType('utcTime', useful.UTCTime()),
    namedtype.NamedType('generalTime', useful.GeneralizedTime())
)


class AttributeValue(univ.Any):
    pass


certificateAttributesMap = {}


class AttributeTypeAndValue(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', AttributeType()),
        namedtype.NamedType(
            'value', AttributeValue(),
            openType=opentype.OpenType('type', certificateAttributesMap)
        )
    )


class RelativeDistinguishedName(univ.SetOf):
    pass


RelativeDistinguishedName.componentType = AttributeTypeAndValue()
RelativeDistinguishedName.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class RDNSequence(univ.SequenceOf):
    pass


RDNSequence.componentType = RelativeDistinguishedName()


class Name(univ.Choice):
    pass


Name.componentType = namedtype.NamedTypes(
    namedtype.NamedType('rdnSequence', RDNSequence())
)


class TBSCertList(univ.Sequence):
    pass


TBSCertList.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('version', Version()),
    namedtype.NamedType('signature', AlgorithmIdentifier()),
    namedtype.NamedType('issuer', Name()),
    namedtype.NamedType('thisUpdate', Time()),
    namedtype.OptionalNamedType('nextUpdate', Time()),
    namedtype.OptionalNamedType(
        'revokedCertificates', univ.SequenceOf(
            componentType=univ.Sequence(
                componentType=namedtype.NamedTypes(
                    namedtype.NamedType('userCertificate', CertificateSerialNumber()),
                    namedtype.NamedType('revocationDate', Time()),
                    namedtype.OptionalNamedType('crlEntryExtensions', Extensions())
                )
            )
        )
    ),
    namedtype.OptionalNamedType(
        'crlExtensions', Extensions().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class CertificateList(univ.Sequence):
    pass


CertificateList.componentType = namedtype.NamedTypes(
    namedtype.NamedType('tbsCertList', TBSCertList()),
    namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
    namedtype.NamedType('signature', univ.BitString())
)


class PhysicalDeliveryOfficeName(PDSParameter):
    pass


ub_extension_attributes = univ.Integer(256)

certificateExtensionsMap = {
}

oraddressExtensionAttributeMap = {
}


class ExtensionAttribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType(
            'extension-attribute-type',
            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, ub_extension_attributes)).subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType(
            'extension-attribute-value',
            univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)),
            openType=opentype.OpenType('extension-attribute-type', oraddressExtensionAttributeMap))
    )

id_qt = _buildOid(id_pkix, 2)

id_qt_cps = _buildOid(id_qt, 1)

id_at_stateOrProvinceName = _buildOid(id_at, 8)

id_at_title = _buildOid(id_at, 12)

id_at_serialNumber = _buildOid(id_at, 5)


class X520dnQualifier(char.PrintableString):
    pass


class PosteRestanteAddress(PDSParameter):
    pass


poste_restante_address = univ.Integer(19)


class UniqueIdentifier(univ.BitString):
    pass


class Validity(univ.Sequence):
    pass


Validity.componentType = namedtype.NamedTypes(
    namedtype.NamedType('notBefore', Time()),
    namedtype.NamedType('notAfter', Time())
)


class SubjectPublicKeyInfo(univ.Sequence):
    pass


SubjectPublicKeyInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('algorithm', AlgorithmIdentifier()),
    namedtype.NamedType('subjectPublicKey', univ.BitString())
)


class TBSCertificate(univ.Sequence):
    pass


TBSCertificate.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version',
                                 Version().subtype(explicitTag=tag.Tag(tag.tagClassContext,
                                                                       tag.tagFormatSimple, 0)).subtype(value="v1")),
    namedtype.NamedType('serialNumber', CertificateSerialNumber()),
    namedtype.NamedType('signature', AlgorithmIdentifier()),
    namedtype.NamedType('issuer', Name()),
    namedtype.NamedType('validity', Validity()),
    namedtype.NamedType('subject', Name()),
    namedtype.NamedType('subjectPublicKeyInfo', SubjectPublicKeyInfo()),
    namedtype.OptionalNamedType('issuerUniqueID', UniqueIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('subjectUniqueID', UniqueIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('extensions',
                                Extensions().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)

physical_delivery_office_name = univ.Integer(10)

ub_name = univ.Integer(32768)


class X520name(univ.Choice):
    pass


X520name.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
    namedtype.NamedType('printableString',
                        char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
    namedtype.NamedType('universalString',
                        char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name)))
)

id_at_dnQualifier = _buildOid(id_at, 46)

ub_serial_number = univ.Integer(64)

ub_pseudonym = univ.Integer(128)

pkcs_9 = _buildOid(1, 2, 840, 113549, 1, 9)


class X121Address(char.NumericString):
    pass


X121Address.subtypeSpec = constraint.ValueSizeConstraint(1, ub_x121_address_length)


class NetworkAddress(X121Address):
    pass


ub_integer_options = univ.Integer(256)

id_at_commonName = _buildOid(id_at, 3)

ub_organization_name_length = univ.Integer(64)

id_ad_ocsp = _buildOid(id_ad, 1)

ub_country_name_numeric_length = univ.Integer(3)

ub_country_name_alpha_length = univ.Integer(2)


class PhysicalDeliveryCountryName(univ.Choice):
    pass


PhysicalDeliveryCountryName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('x121-dcc-code', char.NumericString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_numeric_length, ub_country_name_numeric_length))),
    namedtype.NamedType('iso-3166-alpha2-code', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_alpha_length, ub_country_name_alpha_length)))
)

id_emailAddress = _buildOid(pkcs_9, 1)

common_name = univ.Integer(1)


class X520Pseudonym(univ.Choice):
    pass


X520Pseudonym.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_pseudonym))),
    namedtype.NamedType('printableString',
                        char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_pseudonym))),
    namedtype.NamedType('universalString',
                        char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_pseudonym))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_pseudonym))),
    namedtype.NamedType('bmpString',
                        char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_pseudonym)))
)

ub_domain_name_length = univ.Integer(16)


class AdministrationDomainName(univ.Choice):
    pass


AdministrationDomainName.tagSet = univ.Choice.tagSet.tagExplicitly(
    tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 2))
AdministrationDomainName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('numeric', char.NumericString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(0, ub_domain_name_length))),
    namedtype.NamedType('printable', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(0, ub_domain_name_length)))
)


class PresentationAddress(univ.Sequence):
    pass


PresentationAddress.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('pSelector', univ.OctetString().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('sSelector', univ.OctetString().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('tSelector', univ.OctetString().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.NamedType('nAddresses', univ.SetOf(componentType=univ.OctetString()).subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)


class ExtendedNetworkAddress(univ.Choice):
    pass


ExtendedNetworkAddress.componentType = namedtype.NamedTypes(
    namedtype.NamedType(
        'e163-4-address', univ.Sequence(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType('number', char.NumericString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_e163_4_number_length)).subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
                namedtype.OptionalNamedType('sub-address', char.NumericString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_e163_4_sub_address_length)).subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
            )
        )
    ),
    namedtype.NamedType('psap-address', PresentationAddress().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
)


class TeletexOrganizationName(char.TeletexString):
    pass


TeletexOrganizationName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_organization_name_length)

ub_terminal_id_length = univ.Integer(24)


class TerminalIdentifier(char.PrintableString):
    pass


TerminalIdentifier.subtypeSpec = constraint.ValueSizeConstraint(1, ub_terminal_id_length)

id_ad_caIssuers = _buildOid(id_ad, 2)

id_at_countryName = _buildOid(id_at, 6)


class StreetAddress(PDSParameter):
    pass


postal_code = univ.Integer(9)

id_at_givenName = _buildOid(id_at, 42)

ub_title = univ.Integer(64)


class ExtensionAttributes(univ.SetOf):
    pass


ExtensionAttributes.componentType = ExtensionAttribute()
ExtensionAttributes.sizeSpec = constraint.ValueSizeConstraint(1, ub_extension_attributes)

ub_emailaddress_length = univ.Integer(255)

id_ad_caRepository = _buildOid(id_ad, 5)


class ExtensionORAddressComponents(PDSParameter):
    pass


ub_organizational_unit_name = univ.Integer(64)


class X520OrganizationalUnitName(univ.Choice):
    pass


X520OrganizationalUnitName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organizational_unit_name))),
    namedtype.NamedType('printableString', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organizational_unit_name))),
    namedtype.NamedType('universalString', char.UniversalString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organizational_unit_name))),
    namedtype.NamedType('utf8String', char.UTF8String().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organizational_unit_name))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_organizational_unit_name)))
)


class LocalPostalAttributes(PDSParameter):
    pass


teletex_organizational_unit_names = univ.Integer(5)


class X520Title(univ.Choice):
    pass


X520Title.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
    namedtype.NamedType('printableString',
                        char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
    namedtype.NamedType('universalString',
                        char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title)))
)

id_at_localityName = _buildOid(id_at, 7)

id_at_initials = _buildOid(id_at, 43)

ub_state_name = univ.Integer(128)


class X520StateOrProvinceName(univ.Choice):
    pass


X520StateOrProvinceName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
    namedtype.NamedType('printableString',
                        char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
    namedtype.NamedType('universalString',
                        char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
    namedtype.NamedType('bmpString',
                        char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name)))
)

physical_delivery_organization_name = univ.Integer(14)

id_at_surname = _buildOid(id_at, 4)


class X520countryName(char.PrintableString):
    pass


X520countryName.subtypeSpec = constraint.ValueSizeConstraint(2, 2)

physical_delivery_office_number = univ.Integer(11)

id_qt_unotice = _buildOid(id_qt, 2)


class X520SerialNumber(char.PrintableString):
    pass


X520SerialNumber.subtypeSpec = constraint.ValueSizeConstraint(1, ub_serial_number)


class Attribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', AttributeType()),
        namedtype.NamedType('values',
                            univ.SetOf(componentType=AttributeValue()),
                            openType=opentype.OpenType('type', certificateAttributesMap))
    )

ub_common_name = univ.Integer(64)

id_pe = _buildOid(id_pkix, 1)


class ExtensionPhysicalDeliveryAddressComponents(PDSParameter):
    pass


class EmailAddress(char.IA5String):
    pass


EmailAddress.subtypeSpec = constraint.ValueSizeConstraint(1, ub_emailaddress_length)

id_at_organizationName = _buildOid(id_at, 10)

post_office_box_address = univ.Integer(18)


class BuiltInDomainDefinedAttribute(univ.Sequence):
    pass


BuiltInDomainDefinedAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('type', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_type_length))),
    namedtype.NamedType('value', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_value_length)))
)


class BuiltInDomainDefinedAttributes(univ.SequenceOf):
    pass


BuiltInDomainDefinedAttributes.componentType = BuiltInDomainDefinedAttribute()
BuiltInDomainDefinedAttributes.sizeSpec = constraint.ValueSizeConstraint(1, ub_domain_defined_attributes)

id_at_pseudonym = _buildOid(id_at, 65)

id_domainComponent = _buildOid(0, 9, 2342, 19200300, 100, 1, 25)


class X520CommonName(univ.Choice):
    pass


X520CommonName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
    namedtype.NamedType('printableString',
                        char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
    namedtype.NamedType('universalString',
                        char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
    namedtype.NamedType('utf8String',
                        char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
    namedtype.NamedType('bmpString',
                        char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name)))
)

extension_OR_address_components = univ.Integer(12)

ub_organizational_units = univ.Integer(4)

teletex_personal_name = univ.Integer(4)

ub_numeric_user_id_length = univ.Integer(32)

ub_common_name_length = univ.Integer(64)


class TeletexCommonName(char.TeletexString):
    pass


TeletexCommonName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_common_name_length)


class PhysicalDeliveryOrganizationName(PDSParameter):
    pass


extension_physical_delivery_address_components = univ.Integer(15)


class NumericUserIdentifier(char.NumericString):
    pass


NumericUserIdentifier.subtypeSpec = constraint.ValueSizeConstraint(1, ub_numeric_user_id_length)


class CountryName(univ.Choice):
    pass


CountryName.tagSet = univ.Choice.tagSet.tagExplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 1))
CountryName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('x121-dcc-code', char.NumericString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_numeric_length, ub_country_name_numeric_length))),
    namedtype.NamedType('iso-3166-alpha2-code', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_alpha_length, ub_country_name_alpha_length)))
)


class OrganizationName(char.PrintableString):
    pass


OrganizationName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_organization_name_length)


class OrganizationalUnitNames(univ.SequenceOf):
    pass


OrganizationalUnitNames.componentType = OrganizationalUnitName()
OrganizationalUnitNames.sizeSpec = constraint.ValueSizeConstraint(1, ub_organizational_units)


class PrivateDomainName(univ.Choice):
    pass


PrivateDomainName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('numeric', char.NumericString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_name_length))),
    namedtype.NamedType('printable', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_name_length)))
)


class BuiltInStandardAttributes(univ.Sequence):
    pass


BuiltInStandardAttributes.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('country-name', CountryName()),
    namedtype.OptionalNamedType('administration-domain-name', AdministrationDomainName()),
    namedtype.OptionalNamedType('network-address', NetworkAddress().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('terminal-identifier', TerminalIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('private-domain-name', PrivateDomainName().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))),
    namedtype.OptionalNamedType('organization-name', OrganizationName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
    namedtype.OptionalNamedType('numeric-user-identifier', NumericUserIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))),
    namedtype.OptionalNamedType('personal-name', PersonalName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 5))),
    namedtype.OptionalNamedType('organizational-unit-names', OrganizationalUnitNames().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 6)))
)


class ORAddress(univ.Sequence):
    pass


ORAddress.componentType = namedtype.NamedTypes(
    namedtype.NamedType('built-in-standard-attributes', BuiltInStandardAttributes()),
    namedtype.OptionalNamedType('built-in-domain-defined-attributes', BuiltInDomainDefinedAttributes()),
    namedtype.OptionalNamedType('extension-attributes', ExtensionAttributes())
)


class DistinguishedName(RDNSequence):
    pass


id_ad_timeStamping = _buildOid(id_ad, 3)


class PhysicalDeliveryOfficeNumber(PDSParameter):
    pass


teletex_domain_defined_attributes = univ.Integer(6)


class UniquePostalName(PDSParameter):
    pass


physical_delivery_country_name = univ.Integer(8)

ub_pds_name_length = univ.Integer(16)


class PDSName(char.PrintableString):
    pass


PDSName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_pds_name_length)


class TeletexPersonalName(univ.Set):
    pass


TeletexPersonalName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('surname', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_surname_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('given-name', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_given_name_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('initials', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_initials_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('generation-qualifier', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, ub_generation_qualifier_length)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)

street_address = univ.Integer(17)


class PostOfficeBoxAddress(PDSParameter):
    pass


local_postal_attributes = univ.Integer(21)


class DirectoryString(univ.Choice):
    pass


DirectoryString.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString',
                        char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
    namedtype.NamedType('printableString',
                        char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
    namedtype.NamedType('universalString',
                        char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
    namedtype.NamedType('utf8String', char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX)))
)

teletex_common_name = univ.Integer(2)


class CommonName(char.PrintableString):
    pass


CommonName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_common_name_length)


class Certificate(univ.Sequence):
    pass


Certificate.componentType = namedtype.NamedTypes(
    namedtype.NamedType('tbsCertificate', TBSCertificate()),
    namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
    namedtype.NamedType('signature', univ.BitString())
)


class TeletexOrganizationalUnitName(char.TeletexString):
    pass


TeletexOrganizationalUnitName.subtypeSpec = constraint.ValueSizeConstraint(1, ub_organizational_unit_name_length)

id_at_name = _buildOid(id_at, 41)


class TeletexOrganizationalUnitNames(univ.SequenceOf):
    pass


TeletexOrganizationalUnitNames.componentType = TeletexOrganizationalUnitName()
TeletexOrganizationalUnitNames.sizeSpec = constraint.ValueSizeConstraint(1, ub_organizational_units)

id_ce = _buildOid(2, 5, 29)

id_ce_issuerAltName = _buildOid(id_ce, 18)


class SkipCerts(univ.Integer):
    pass


SkipCerts.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


class CRLReason(univ.Enumerated):
    pass


CRLReason.namedValues = namedval.NamedValues(
    ('unspecified', 0),
    ('keyCompromise', 1),
    ('cACompromise', 2),
    ('affiliationChanged', 3),
    ('superseded', 4),
    ('cessationOfOperation', 5),
    ('certificateHold', 6),
    ('removeFromCRL', 8),
    ('privilegeWithdrawn', 9),
    ('aACompromise', 10)
)


class PrivateKeyUsagePeriod(univ.Sequence):
    pass


PrivateKeyUsagePeriod.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('notBefore', useful.GeneralizedTime().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('notAfter', useful.GeneralizedTime().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


anotherNameMap = {

}


class AnotherName(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type-id', univ.ObjectIdentifier()),
        namedtype.NamedType(
            'value',
            univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)),
            openType=opentype.OpenType('type-id', anotherNameMap)
        )
    )


class EDIPartyName(univ.Sequence):
    pass


EDIPartyName.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('nameAssigner', DirectoryString().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('partyName', DirectoryString().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
)


class GeneralName(univ.Choice):
    pass


GeneralName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('otherName',
                        AnotherName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('rfc822Name',
                        char.IA5String().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('dNSName',
                        char.IA5String().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.NamedType('x400Address',
                        ORAddress().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
    namedtype.NamedType('directoryName',
                        Name().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))),
    namedtype.NamedType('ediPartyName',
                        EDIPartyName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 5))),
    namedtype.NamedType('uniformResourceIdentifier',
                        char.IA5String().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 6))),
    namedtype.NamedType('iPAddress',
                        univ.OctetString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 7))),
    namedtype.NamedType('registeredID', univ.ObjectIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 8)))
)


class BaseDistance(univ.Integer):
    pass


BaseDistance.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


class GeneralSubtree(univ.Sequence):
    pass


GeneralSubtree.componentType = namedtype.NamedTypes(
    namedtype.NamedType('base', GeneralName()),
    namedtype.DefaultedNamedType('minimum', BaseDistance().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)).subtype(value=0)),
    namedtype.OptionalNamedType('maximum', BaseDistance().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class GeneralNames(univ.SequenceOf):
    pass


GeneralNames.componentType = GeneralName()
GeneralNames.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class DistributionPointName(univ.Choice):
    pass


DistributionPointName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('fullName',
                        GeneralNames().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('nameRelativeToCRLIssuer', RelativeDistinguishedName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class ReasonFlags(univ.BitString):
    pass


ReasonFlags.namedValues = namedval.NamedValues(
    ('unused', 0),
    ('keyCompromise', 1),
    ('cACompromise', 2),
    ('affiliationChanged', 3),
    ('superseded', 4),
    ('cessationOfOperation', 5),
    ('certificateHold', 6),
    ('privilegeWithdrawn', 7),
    ('aACompromise', 8)
)


class IssuingDistributionPoint(univ.Sequence):
    pass


IssuingDistributionPoint.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('distributionPoint', DistributionPointName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.DefaultedNamedType('onlyContainsUserCerts', univ.Boolean().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)).subtype(value=0)),
    namedtype.DefaultedNamedType('onlyContainsCACerts', univ.Boolean().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)).subtype(value=0)),
    namedtype.OptionalNamedType('onlySomeReasons', ReasonFlags().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
    namedtype.DefaultedNamedType('indirectCRL', univ.Boolean().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4)).subtype(value=0)),
    namedtype.DefaultedNamedType('onlyContainsAttributeCerts', univ.Boolean().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5)).subtype(value=0))
)

id_ce_certificatePolicies = _buildOid(id_ce, 32)

id_kp_emailProtection = _buildOid(id_kp, 4)


class AccessDescription(univ.Sequence):
    pass


AccessDescription.componentType = namedtype.NamedTypes(
    namedtype.NamedType('accessMethod', univ.ObjectIdentifier()),
    namedtype.NamedType('accessLocation', GeneralName())
)


class IssuerAltName(GeneralNames):
    pass


id_ce_cRLDistributionPoints = _buildOid(id_ce, 31)

holdInstruction = _buildOid(2, 2, 840, 10040, 2)

id_holdinstruction_callissuer = _buildOid(holdInstruction, 2)

id_ce_subjectDirectoryAttributes = _buildOid(id_ce, 9)

id_ce_issuingDistributionPoint = _buildOid(id_ce, 28)


class DistributionPoint(univ.Sequence):
    pass


DistributionPoint.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('distributionPoint', DistributionPointName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('reasons', ReasonFlags().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('cRLIssuer', GeneralNames().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)


class CRLDistributionPoints(univ.SequenceOf):
    pass


CRLDistributionPoints.componentType = DistributionPoint()
CRLDistributionPoints.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class GeneralSubtrees(univ.SequenceOf):
    pass


GeneralSubtrees.componentType = GeneralSubtree()
GeneralSubtrees.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class NameConstraints(univ.Sequence):
    pass


NameConstraints.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('permittedSubtrees', GeneralSubtrees().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('excludedSubtrees', GeneralSubtrees().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class SubjectDirectoryAttributes(univ.SequenceOf):
    pass


SubjectDirectoryAttributes.componentType = Attribute()
SubjectDirectoryAttributes.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

id_kp_OCSPSigning = _buildOid(id_kp, 9)

id_kp_timeStamping = _buildOid(id_kp, 8)


class DisplayText(univ.Choice):
    pass


DisplayText.componentType = namedtype.NamedTypes(
    namedtype.NamedType('ia5String', char.IA5String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200))),
    namedtype.NamedType('visibleString',
                        char.VisibleString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200))),
    namedtype.NamedType('utf8String', char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200)))
)


class NoticeReference(univ.Sequence):
    pass


NoticeReference.componentType = namedtype.NamedTypes(
    namedtype.NamedType('organization', DisplayText()),
    namedtype.NamedType('noticeNumbers', univ.SequenceOf(componentType=univ.Integer()))
)


class UserNotice(univ.Sequence):
    pass


UserNotice.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('noticeRef', NoticeReference()),
    namedtype.OptionalNamedType('explicitText', DisplayText())
)


class PolicyQualifierId(univ.ObjectIdentifier):
    pass


policyQualifierInfoMap = {

}


class PolicyQualifierInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('policyQualifierId', PolicyQualifierId()),
        namedtype.NamedType(
            'qualifier', univ.Any(),
            openType=opentype.OpenType('policyQualifierId', policyQualifierInfoMap)
        )
    )


class CertPolicyId(univ.ObjectIdentifier):
    pass


class PolicyInformation(univ.Sequence):
    pass


PolicyInformation.componentType = namedtype.NamedTypes(
    namedtype.NamedType('policyIdentifier', CertPolicyId()),
    namedtype.OptionalNamedType('policyQualifiers', univ.SequenceOf(componentType=PolicyQualifierInfo()))
)


class CertificatePolicies(univ.SequenceOf):
    pass


CertificatePolicies.componentType = PolicyInformation()
CertificatePolicies.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class SubjectAltName(GeneralNames):
    pass


id_ce_basicConstraints = _buildOid(id_ce, 19)

id_ce_authorityKeyIdentifier = _buildOid(id_ce, 35)

id_kp_codeSigning = _buildOid(id_kp, 3)


class BasicConstraints(univ.Sequence):
    pass


BasicConstraints.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('cA', univ.Boolean().subtype(value=0)),
    namedtype.OptionalNamedType('pathLenConstraint',
                                univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, MAX)))
)

id_ce_certificateIssuer = _buildOid(id_ce, 29)


class PolicyMappings(univ.SequenceOf):
    pass


PolicyMappings.componentType = univ.Sequence(
    componentType=namedtype.NamedTypes(
        namedtype.NamedType('issuerDomainPolicy', CertPolicyId()),
        namedtype.NamedType('subjectDomainPolicy', CertPolicyId())
    )
)

PolicyMappings.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class InhibitAnyPolicy(SkipCerts):
    pass


anyPolicy = _buildOid(id_ce_certificatePolicies, 0)


class CRLNumber(univ.Integer):
    pass


CRLNumber.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


class BaseCRLNumber(CRLNumber):
    pass


id_ce_nameConstraints = _buildOid(id_ce, 30)

id_kp_serverAuth = _buildOid(id_kp, 1)

id_ce_freshestCRL = _buildOid(id_ce, 46)

id_ce_cRLReasons = _buildOid(id_ce, 21)

id_ce_extKeyUsage = _buildOid(id_ce, 37)


class KeyIdentifier(univ.OctetString):
    pass


class AuthorityKeyIdentifier(univ.Sequence):
    pass


AuthorityKeyIdentifier.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('keyIdentifier', KeyIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('authorityCertIssuer', GeneralNames().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('authorityCertSerialNumber', CertificateSerialNumber().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)


class FreshestCRL(CRLDistributionPoints):
    pass


id_ce_policyConstraints = _buildOid(id_ce, 36)

id_pe_authorityInfoAccess = _buildOid(id_pe, 1)


class AuthorityInfoAccessSyntax(univ.SequenceOf):
    pass


AuthorityInfoAccessSyntax.componentType = AccessDescription()
AuthorityInfoAccessSyntax.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

id_holdinstruction_none = _buildOid(holdInstruction, 1)


class CPSuri(char.IA5String):
    pass


id_pe_subjectInfoAccess = _buildOid(id_pe, 11)


class SubjectKeyIdentifier(KeyIdentifier):
    pass


id_ce_subjectAltName = _buildOid(id_ce, 17)


class KeyPurposeId(univ.ObjectIdentifier):
    pass


class ExtKeyUsageSyntax(univ.SequenceOf):
    pass


ExtKeyUsageSyntax.componentType = KeyPurposeId()
ExtKeyUsageSyntax.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class HoldInstructionCode(univ.ObjectIdentifier):
    pass


id_ce_deltaCRLIndicator = _buildOid(id_ce, 27)

id_ce_keyUsage = _buildOid(id_ce, 15)

id_ce_holdInstructionCode = _buildOid(id_ce, 23)


class SubjectInfoAccessSyntax(univ.SequenceOf):
    pass


SubjectInfoAccessSyntax.componentType = AccessDescription()
SubjectInfoAccessSyntax.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class InvalidityDate(useful.GeneralizedTime):
    pass


class KeyUsage(univ.BitString):
    pass


KeyUsage.namedValues = namedval.NamedValues(
    ('digitalSignature', 0),
    ('nonRepudiation', 1),
    ('keyEncipherment', 2),
    ('dataEncipherment', 3),
    ('keyAgreement', 4),
    ('keyCertSign', 5),
    ('cRLSign', 6),
    ('encipherOnly', 7),
    ('decipherOnly', 8)
)

id_ce_invalidityDate = _buildOid(id_ce, 24)

id_ce_policyMappings = _buildOid(id_ce, 33)

anyExtendedKeyUsage = _buildOid(id_ce_extKeyUsage, 0)

id_ce_privateKeyUsagePeriod = _buildOid(id_ce, 16)

id_ce_cRLNumber = _buildOid(id_ce, 20)


class CertificateIssuer(GeneralNames):
    pass


id_holdinstruction_reject = _buildOid(holdInstruction, 3)


class PolicyConstraints(univ.Sequence):
    pass


PolicyConstraints.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('requireExplicitPolicy',
                                SkipCerts().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('inhibitPolicyMapping',
                                SkipCerts().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)

id_kp_clientAuth = _buildOid(id_kp, 2)

id_ce_subjectKeyIdentifier = _buildOid(id_ce, 14)

id_ce_inhibitAnyPolicy = _buildOid(id_ce, 54)

# map of ORAddress ExtensionAttribute type to ExtensionAttribute value

_oraddressExtensionAttributeMapUpdate = {
    common_name: CommonName(),
    teletex_common_name: TeletexCommonName(),
    teletex_organization_name: TeletexOrganizationName(),
    teletex_personal_name: TeletexPersonalName(),
    teletex_organizational_unit_names: TeletexOrganizationalUnitNames(),
    pds_name: PDSName(),
    physical_delivery_country_name: PhysicalDeliveryCountryName(),
    postal_code: PostalCode(),
    physical_delivery_office_name: PhysicalDeliveryOfficeName(),
    physical_delivery_office_number: PhysicalDeliveryOfficeNumber(),
    extension_OR_address_components: ExtensionORAddressComponents(),
    physical_delivery_personal_name: PhysicalDeliveryPersonalName(),
    physical_delivery_organization_name: PhysicalDeliveryOrganizationName(),
    extension_physical_delivery_address_components: ExtensionPhysicalDeliveryAddressComponents(),
    unformatted_postal_address: UnformattedPostalAddress(),
    street_address: StreetAddress(),
    post_office_box_address: PostOfficeBoxAddress(),
    poste_restante_address: PosteRestanteAddress(),
    unique_postal_name: UniquePostalName(),
    local_postal_attributes: LocalPostalAttributes(),
    extended_network_address: ExtendedNetworkAddress(),
    terminal_type: TerminalType(),
    teletex_domain_defined_attributes: TeletexDomainDefinedAttributes(),
}

oraddressExtensionAttributeMap.update(_oraddressExtensionAttributeMapUpdate)


# map of AttributeType -> AttributeValue

_certificateAttributesMapUpdate = {
    id_at_name: X520name(),
    id_at_surname: X520name(),
    id_at_givenName: X520name(),
    id_at_initials: X520name(),
    id_at_generationQualifier: X520name(),
    id_at_commonName: X520CommonName(),
    id_at_localityName: X520LocalityName(),
    id_at_stateOrProvinceName: X520StateOrProvinceName(),
    id_at_organizationName: X520OrganizationName(),
    id_at_organizationalUnitName: X520OrganizationalUnitName(),
    id_at_title: X520Title(),
    id_at_dnQualifier: X520dnQualifier(),
    id_at_countryName: X520countryName(),
    id_at_serialNumber: X520SerialNumber(),
    id_at_pseudonym: X520Pseudonym(),
    id_domainComponent: DomainComponent(),
    id_emailAddress: EmailAddress(),
}

certificateAttributesMap.update(_certificateAttributesMapUpdate)


# map of Certificate Extension OIDs to Extensions

_certificateExtensionsMap = {
    id_ce_authorityKeyIdentifier: AuthorityKeyIdentifier(),
    id_ce_subjectKeyIdentifier: SubjectKeyIdentifier(),
    id_ce_keyUsage: KeyUsage(),
    id_ce_privateKeyUsagePeriod: PrivateKeyUsagePeriod(),
    id_ce_certificatePolicies: CertificatePolicies(),
    id_ce_policyMappings: PolicyMappings(),
    id_ce_subjectAltName: SubjectAltName(),
    id_ce_issuerAltName: IssuerAltName(),
    id_ce_subjectDirectoryAttributes: SubjectDirectoryAttributes(),
    id_ce_basicConstraints: BasicConstraints(),
    id_ce_nameConstraints: NameConstraints(),
    id_ce_policyConstraints: PolicyConstraints(),
    id_ce_extKeyUsage: ExtKeyUsageSyntax(),
    id_ce_cRLDistributionPoints: CRLDistributionPoints(),
    id_pe_authorityInfoAccess: AuthorityInfoAccessSyntax(),
    id_ce_cRLNumber: univ.Integer(),
    id_ce_deltaCRLIndicator: BaseCRLNumber(),
    id_ce_issuingDistributionPoint: IssuingDistributionPoint(),
    id_ce_cRLReasons: CRLReason(),
    id_ce_holdInstructionCode: univ.ObjectIdentifier(),
    id_ce_invalidityDate: useful.GeneralizedTime(),
    id_ce_certificateIssuer: GeneralNames(),
}

certificateExtensionsMap.update(_certificateExtensionsMap)
