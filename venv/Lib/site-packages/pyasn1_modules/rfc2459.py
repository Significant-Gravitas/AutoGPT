#
# This file is part of pyasn1-modules software.
#
# Updated by Russ Housley to resolve the TODO regarding the Certificate
#   Policies Certificate Extension.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# X.509 message syntax
#
# ASN.1 source from:
# http://www.trl.ibm.com/projects/xml/xss4j/data/asn1/grammars/x509.asn
# http://www.ietf.org/rfc/rfc2459.txt
#
# Sample captures from:
# http://wiki.wireshark.org/SampleCaptures/
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

#
# PKIX1Explicit88
#

# Upper Bounds
ub_name = univ.Integer(32768)
ub_common_name = univ.Integer(64)
ub_locality_name = univ.Integer(128)
ub_state_name = univ.Integer(128)
ub_organization_name = univ.Integer(64)
ub_organizational_unit_name = univ.Integer(64)
ub_title = univ.Integer(64)
ub_match = univ.Integer(128)
ub_emailaddress_length = univ.Integer(128)
ub_common_name_length = univ.Integer(64)
ub_country_name_alpha_length = univ.Integer(2)
ub_country_name_numeric_length = univ.Integer(3)
ub_domain_defined_attributes = univ.Integer(4)
ub_domain_defined_attribute_type_length = univ.Integer(8)
ub_domain_defined_attribute_value_length = univ.Integer(128)
ub_domain_name_length = univ.Integer(16)
ub_extension_attributes = univ.Integer(256)
ub_e163_4_number_length = univ.Integer(15)
ub_e163_4_sub_address_length = univ.Integer(40)
ub_generation_qualifier_length = univ.Integer(3)
ub_given_name_length = univ.Integer(16)
ub_initials_length = univ.Integer(5)
ub_integer_options = univ.Integer(256)
ub_numeric_user_id_length = univ.Integer(32)
ub_organization_name_length = univ.Integer(64)
ub_organizational_unit_name_length = univ.Integer(32)
ub_organizational_units = univ.Integer(4)
ub_pds_name_length = univ.Integer(16)
ub_pds_parameter_length = univ.Integer(30)
ub_pds_physical_address_lines = univ.Integer(6)
ub_postal_code_length = univ.Integer(16)
ub_surname_length = univ.Integer(40)
ub_terminal_id_length = univ.Integer(24)
ub_unformatted_address_length = univ.Integer(180)
ub_x121_address_length = univ.Integer(16)


class UniversalString(char.UniversalString):
    pass


class BMPString(char.BMPString):
    pass


class UTF8String(char.UTF8String):
    pass


id_pkix = univ.ObjectIdentifier('1.3.6.1.5.5.7')
id_pe = univ.ObjectIdentifier('1.3.6.1.5.5.7.1')
id_qt = univ.ObjectIdentifier('1.3.6.1.5.5.7.2')
id_kp = univ.ObjectIdentifier('1.3.6.1.5.5.7.3')
id_ad = univ.ObjectIdentifier('1.3.6.1.5.5.7.48')

id_qt_cps = univ.ObjectIdentifier('1.3.6.1.5.5.7.2.1')
id_qt_unotice = univ.ObjectIdentifier('1.3.6.1.5.5.7.2.2')

id_ad_ocsp = univ.ObjectIdentifier('1.3.6.1.5.5.7.48.1')
id_ad_caIssuers = univ.ObjectIdentifier('1.3.6.1.5.5.7.48.2')




id_at = univ.ObjectIdentifier('2.5.4')
id_at_name = univ.ObjectIdentifier('2.5.4.41')
# preserve misspelled variable for compatibility
id_at_sutname = id_at_surname = univ.ObjectIdentifier('2.5.4.4')
id_at_givenName = univ.ObjectIdentifier('2.5.4.42')
id_at_initials = univ.ObjectIdentifier('2.5.4.43')
id_at_generationQualifier = univ.ObjectIdentifier('2.5.4.44')


class X520name(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString',
                            char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
        namedtype.NamedType('printableString',
                            char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
        namedtype.NamedType('universalString',
                            char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
        namedtype.NamedType('utf8String',
                            char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name))),
        namedtype.NamedType('bmpString',
                            char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_name)))
    )


id_at_commonName = univ.ObjectIdentifier('2.5.4.3')


class X520CommonName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
        namedtype.NamedType('printableString', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
        namedtype.NamedType('universalString', char.UniversalString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
        namedtype.NamedType('utf8String',
                            char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name))),
        namedtype.NamedType('bmpString',
                            char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_common_name)))
    )


id_at_localityName = univ.ObjectIdentifier('2.5.4.7')


class X520LocalityName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
        namedtype.NamedType('printableString', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
        namedtype.NamedType('universalString', char.UniversalString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
        namedtype.NamedType('utf8String',
                            char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name))),
        namedtype.NamedType('bmpString',
                            char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_locality_name)))
    )


id_at_stateOrProvinceName = univ.ObjectIdentifier('2.5.4.8')


class X520StateOrProvinceName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString',
                            char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
        namedtype.NamedType('printableString', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
        namedtype.NamedType('universalString', char.UniversalString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
        namedtype.NamedType('utf8String',
                            char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name))),
        namedtype.NamedType('bmpString',
                            char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_state_name)))
    )


id_at_organizationName = univ.ObjectIdentifier('2.5.4.10')


class X520OrganizationName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
        namedtype.NamedType('printableString', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
        namedtype.NamedType('universalString', char.UniversalString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
        namedtype.NamedType('utf8String', char.UTF8String().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name))),
        namedtype.NamedType('bmpString', char.BMPString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_organization_name)))
    )


id_at_organizationalUnitName = univ.ObjectIdentifier('2.5.4.11')


class X520OrganizationalUnitName(univ.Choice):
    componentType = namedtype.NamedTypes(
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


id_at_title = univ.ObjectIdentifier('2.5.4.12')


class X520Title(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString',
                            char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
        namedtype.NamedType('printableString',
                            char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
        namedtype.NamedType('universalString',
                            char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
        namedtype.NamedType('utf8String',
                            char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title))),
        namedtype.NamedType('bmpString',
                            char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_title)))
    )


id_at_dnQualifier = univ.ObjectIdentifier('2.5.4.46')


class X520dnQualifier(char.PrintableString):
    pass


id_at_countryName = univ.ObjectIdentifier('2.5.4.6')


class X520countryName(char.PrintableString):
    subtypeSpec = char.PrintableString.subtypeSpec + constraint.ValueSizeConstraint(2, 2)


pkcs_9 = univ.ObjectIdentifier('1.2.840.113549.1.9')

emailAddress = univ.ObjectIdentifier('1.2.840.113549.1.9.1')


class Pkcs9email(char.IA5String):
    subtypeSpec = char.IA5String.subtypeSpec + constraint.ValueSizeConstraint(1, ub_emailaddress_length)


# ----

class DSAPrivateKey(univ.Sequence):
    """PKIX compliant DSA private key structure"""
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', univ.Integer(namedValues=namedval.NamedValues(('v1', 0)))),
        namedtype.NamedType('p', univ.Integer()),
        namedtype.NamedType('q', univ.Integer()),
        namedtype.NamedType('g', univ.Integer()),
        namedtype.NamedType('public', univ.Integer()),
        namedtype.NamedType('private', univ.Integer())
    )


# ----


class DirectoryString(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('teletexString',
                            char.TeletexString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
        namedtype.NamedType('printableString',
                            char.PrintableString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
        namedtype.NamedType('universalString',
                            char.UniversalString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
        namedtype.NamedType('utf8String',
                            char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
        namedtype.NamedType('bmpString', char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
        namedtype.NamedType('ia5String', char.IA5String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX)))
        # hm, this should not be here!? XXX
    )


# certificate and CRL specific structures begin here

class AlgorithmIdentifier(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('algorithm', univ.ObjectIdentifier()),
        namedtype.OptionalNamedType('parameters', univ.Any())
    )



# Algorithm OIDs and parameter structures

pkcs_1 = univ.ObjectIdentifier('1.2.840.113549.1.1')
rsaEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.1')
md2WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.2')
md5WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.4')
sha1WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.5')
id_dsa_with_sha1 = univ.ObjectIdentifier('1.2.840.10040.4.3')


class Dss_Sig_Value(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('r', univ.Integer()),
        namedtype.NamedType('s', univ.Integer())
    )


dhpublicnumber = univ.ObjectIdentifier('1.2.840.10046.2.1')


class ValidationParms(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('seed', univ.BitString()),
        namedtype.NamedType('pgenCounter', univ.Integer())
    )


class DomainParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('p', univ.Integer()),
        namedtype.NamedType('g', univ.Integer()),
        namedtype.NamedType('q', univ.Integer()),
        namedtype.NamedType('j', univ.Integer()),
        namedtype.OptionalNamedType('validationParms', ValidationParms())
    )


id_dsa = univ.ObjectIdentifier('1.2.840.10040.4.1')


class Dss_Parms(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('p', univ.Integer()),
        namedtype.NamedType('q', univ.Integer()),
        namedtype.NamedType('g', univ.Integer())
    )


# x400 address syntax starts here

teletex_domain_defined_attributes = univ.Integer(6)


class TeletexDomainDefinedAttribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_type_length))),
        namedtype.NamedType('value', char.TeletexString())
    )


class TeletexDomainDefinedAttributes(univ.SequenceOf):
    componentType = TeletexDomainDefinedAttribute()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, ub_domain_defined_attributes)


terminal_type = univ.Integer(23)


class TerminalType(univ.Integer):
    subtypeSpec = univ.Integer.subtypeSpec + constraint.ValueSizeConstraint(0, ub_integer_options)
    namedValues = namedval.NamedValues(
        ('telex', 3),
        ('teletelex', 4),
        ('g3-facsimile', 5),
        ('g4-facsimile', 6),
        ('ia5-terminal', 7),
        ('videotex', 8)
    )


class PresentationAddress(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('pSelector', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('sSelector', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('tSelector', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.OptionalNamedType('nAddresses', univ.SetOf(componentType=univ.OctetString()).subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3),
            subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
    )


extended_network_address = univ.Integer(22)


class E163_4_address(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('number', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_e163_4_number_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('sub-address', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_e163_4_sub_address_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class ExtendedNetworkAddress(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('e163-4-address', E163_4_address()),
        namedtype.NamedType('psap-address', PresentationAddress().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class PDSParameter(univ.Set):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('printable-string', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_parameter_length))),
        namedtype.OptionalNamedType('teletex-string', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_parameter_length)))
    )


local_postal_attributes = univ.Integer(21)


class LocalPostalAttributes(PDSParameter):
    pass


class UniquePostalName(PDSParameter):
    pass


unique_postal_name = univ.Integer(20)

poste_restante_address = univ.Integer(19)


class PosteRestanteAddress(PDSParameter):
    pass


post_office_box_address = univ.Integer(18)


class PostOfficeBoxAddress(PDSParameter):
    pass


street_address = univ.Integer(17)


class StreetAddress(PDSParameter):
    pass


class UnformattedPostalAddress(univ.Set):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('printable-address', univ.SequenceOf(componentType=char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_parameter_length)).subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_pds_physical_address_lines)))),
        namedtype.OptionalNamedType('teletex-string', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_unformatted_address_length)))
    )


physical_delivery_office_name = univ.Integer(10)


class PhysicalDeliveryOfficeName(PDSParameter):
    pass


physical_delivery_office_number = univ.Integer(11)


class PhysicalDeliveryOfficeNumber(PDSParameter):
    pass


extension_OR_address_components = univ.Integer(12)


class ExtensionORAddressComponents(PDSParameter):
    pass


physical_delivery_personal_name = univ.Integer(13)


class PhysicalDeliveryPersonalName(PDSParameter):
    pass


physical_delivery_organization_name = univ.Integer(14)


class PhysicalDeliveryOrganizationName(PDSParameter):
    pass


extension_physical_delivery_address_components = univ.Integer(15)


class ExtensionPhysicalDeliveryAddressComponents(PDSParameter):
    pass


unformatted_postal_address = univ.Integer(16)

postal_code = univ.Integer(9)


class PostalCode(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('numeric-code', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_postal_code_length))),
        namedtype.NamedType('printable-code', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_postal_code_length)))
    )


class PhysicalDeliveryCountryName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('x121-dcc-code', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_numeric_length,
                                                       ub_country_name_numeric_length))),
        namedtype.NamedType('iso-3166-alpha2-code', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_alpha_length, ub_country_name_alpha_length)))
    )


class PDSName(char.PrintableString):
    subtypeSpec = char.PrintableString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_pds_name_length)


physical_delivery_country_name = univ.Integer(8)


class TeletexOrganizationalUnitName(char.TeletexString):
    subtypeSpec = char.TeletexString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_organizational_unit_name_length)


pds_name = univ.Integer(7)

teletex_organizational_unit_names = univ.Integer(5)


class TeletexOrganizationalUnitNames(univ.SequenceOf):
    componentType = TeletexOrganizationalUnitName()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, ub_organizational_units)


teletex_personal_name = univ.Integer(4)


class TeletexPersonalName(univ.Set):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('surname', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_surname_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('given-name', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_given_name_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('initials', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_initials_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.OptionalNamedType('generation-qualifier', char.TeletexString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_generation_qualifier_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
    )


teletex_organization_name = univ.Integer(3)


class TeletexOrganizationName(char.TeletexString):
    subtypeSpec = char.TeletexString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_organization_name_length)


teletex_common_name = univ.Integer(2)


class TeletexCommonName(char.TeletexString):
    subtypeSpec = char.TeletexString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_common_name_length)


class CommonName(char.PrintableString):
    subtypeSpec = char.PrintableString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_common_name_length)


common_name = univ.Integer(1)


class ExtensionAttribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('extension-attribute-type', univ.Integer().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(0, ub_extension_attributes),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('extension-attribute-value',
                            univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class ExtensionAttributes(univ.SetOf):
    componentType = ExtensionAttribute()
    sizeSpec = univ.SetOf.sizeSpec + constraint.ValueSizeConstraint(1, ub_extension_attributes)


class BuiltInDomainDefinedAttribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_type_length))),
        namedtype.NamedType('value', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_defined_attribute_value_length)))
    )


class BuiltInDomainDefinedAttributes(univ.SequenceOf):
    componentType = BuiltInDomainDefinedAttribute()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, ub_domain_defined_attributes)


class OrganizationalUnitName(char.PrintableString):
    subtypeSpec = char.PrintableString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_organizational_unit_name_length)


class OrganizationalUnitNames(univ.SequenceOf):
    componentType = OrganizationalUnitName()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, ub_organizational_units)


class PersonalName(univ.Set):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('surname', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_surname_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('given-name', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_given_name_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('initials', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_initials_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.OptionalNamedType('generation-qualifier', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_generation_qualifier_length),
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
    )


class NumericUserIdentifier(char.NumericString):
    subtypeSpec = char.NumericString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_numeric_user_id_length)


class OrganizationName(char.PrintableString):
    subtypeSpec = char.PrintableString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_organization_name_length)


class PrivateDomainName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('numeric', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_name_length))),
        namedtype.NamedType('printable', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, ub_domain_name_length)))
    )


class TerminalIdentifier(char.PrintableString):
    subtypeSpec = char.PrintableString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_terminal_id_length)


class X121Address(char.NumericString):
    subtypeSpec = char.NumericString.subtypeSpec + constraint.ValueSizeConstraint(1, ub_x121_address_length)


class NetworkAddress(X121Address):
    pass


class AdministrationDomainName(univ.Choice):
    tagSet = univ.Choice.tagSet.tagExplicitly(
        tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 2)
    )
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('numeric', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(0, ub_domain_name_length))),
        namedtype.NamedType('printable', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(0, ub_domain_name_length)))
    )


class CountryName(univ.Choice):
    tagSet = univ.Choice.tagSet.tagExplicitly(
        tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 1)
    )
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('x121-dcc-code', char.NumericString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_numeric_length,
                                                       ub_country_name_numeric_length))),
        namedtype.NamedType('iso-3166-alpha2-code', char.PrintableString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(ub_country_name_alpha_length, ub_country_name_alpha_length)))
    )


class BuiltInStandardAttributes(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('country-name', CountryName()),
        namedtype.OptionalNamedType('administration-domain-name', AdministrationDomainName()),
        namedtype.OptionalNamedType('network-address', NetworkAddress().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('terminal-identifier', TerminalIdentifier().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('private-domain-name', PrivateDomainName().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.OptionalNamedType('organization-name', OrganizationName().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
        namedtype.OptionalNamedType('numeric-user-identifier', NumericUserIdentifier().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))),
        namedtype.OptionalNamedType('personal-name', PersonalName().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5))),
        namedtype.OptionalNamedType('organizational-unit-names', OrganizationalUnitNames().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 6)))
    )


class ORAddress(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('built-in-standard-attributes', BuiltInStandardAttributes()),
        namedtype.OptionalNamedType('built-in-domain-defined-attributes', BuiltInDomainDefinedAttributes()),
        namedtype.OptionalNamedType('extension-attributes', ExtensionAttributes())
    )


#
# PKIX1Implicit88
#

id_ce_invalidityDate = univ.ObjectIdentifier('2.5.29.24')


class InvalidityDate(useful.GeneralizedTime):
    pass


id_holdinstruction_none = univ.ObjectIdentifier('2.2.840.10040.2.1')
id_holdinstruction_callissuer = univ.ObjectIdentifier('2.2.840.10040.2.2')
id_holdinstruction_reject = univ.ObjectIdentifier('2.2.840.10040.2.3')

holdInstruction = univ.ObjectIdentifier('2.2.840.10040.2')

id_ce_holdInstructionCode = univ.ObjectIdentifier('2.5.29.23')


class HoldInstructionCode(univ.ObjectIdentifier):
    pass


id_ce_cRLReasons = univ.ObjectIdentifier('2.5.29.21')


class CRLReason(univ.Enumerated):
    namedValues = namedval.NamedValues(
        ('unspecified', 0),
        ('keyCompromise', 1),
        ('cACompromise', 2),
        ('affiliationChanged', 3),
        ('superseded', 4),
        ('cessationOfOperation', 5),
        ('certificateHold', 6),
        ('removeFromCRL', 8)
    )


id_ce_cRLNumber = univ.ObjectIdentifier('2.5.29.20')


class CRLNumber(univ.Integer):
    subtypeSpec = univ.Integer.subtypeSpec + constraint.ValueSizeConstraint(0, MAX)


class BaseCRLNumber(CRLNumber):
    pass


id_kp_serverAuth = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.1')
id_kp_clientAuth = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.2')
id_kp_codeSigning = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.3')
id_kp_emailProtection = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.4')
id_kp_ipsecEndSystem = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.5')
id_kp_ipsecTunnel = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.6')
id_kp_ipsecUser = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.7')
id_kp_timeStamping = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.8')
id_pe_authorityInfoAccess = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.1')
id_ce_extKeyUsage = univ.ObjectIdentifier('2.5.29.37')


class KeyPurposeId(univ.ObjectIdentifier):
    pass


class ExtKeyUsageSyntax(univ.SequenceOf):
    componentType = KeyPurposeId()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class ReasonFlags(univ.BitString):
    namedValues = namedval.NamedValues(
        ('unused', 0),
        ('keyCompromise', 1),
        ('cACompromise', 2),
        ('affiliationChanged', 3),
        ('superseded', 4),
        ('cessationOfOperation', 5),
        ('certificateHold', 6)
    )


class SkipCerts(univ.Integer):
    subtypeSpec = univ.Integer.subtypeSpec + constraint.ValueSizeConstraint(0, MAX)


id_ce_policyConstraints = univ.ObjectIdentifier('2.5.29.36')


class PolicyConstraints(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('requireExplicitPolicy', SkipCerts().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('inhibitPolicyMapping', SkipCerts().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


id_ce_basicConstraints = univ.ObjectIdentifier('2.5.29.19')


class BasicConstraints(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('cA', univ.Boolean(False)),
        namedtype.OptionalNamedType('pathLenConstraint',
                                    univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, MAX)))
    )


id_ce_subjectDirectoryAttributes = univ.ObjectIdentifier('2.5.29.9')


class EDIPartyName(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('nameAssigner', DirectoryString().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('partyName',
                            DirectoryString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )



id_ce_deltaCRLIndicator = univ.ObjectIdentifier('2.5.29.27')



class BaseDistance(univ.Integer):
    subtypeSpec = univ.Integer.subtypeSpec + constraint.ValueRangeConstraint(0, MAX)


id_ce_cRLDistributionPoints = univ.ObjectIdentifier('2.5.29.31')


id_ce_issuingDistributionPoint = univ.ObjectIdentifier('2.5.29.28')




id_ce_nameConstraints = univ.ObjectIdentifier('2.5.29.30')


class DisplayText(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('visibleString',
                            char.VisibleString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200))),
        namedtype.NamedType('bmpString', char.BMPString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200))),
        namedtype.NamedType('utf8String', char.UTF8String().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 200)))
    )


class NoticeReference(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('organization', DisplayText()),
        namedtype.NamedType('noticeNumbers', univ.SequenceOf(componentType=univ.Integer()))
    )


class UserNotice(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('noticeRef', NoticeReference()),
        namedtype.OptionalNamedType('explicitText', DisplayText())
    )


class CPSuri(char.IA5String):
    pass


class PolicyQualifierId(univ.ObjectIdentifier):
    subtypeSpec = univ.ObjectIdentifier.subtypeSpec + constraint.SingleValueConstraint(id_qt_cps, id_qt_unotice)


class CertPolicyId(univ.ObjectIdentifier):
    pass


class PolicyQualifierInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('policyQualifierId', PolicyQualifierId()),
        namedtype.NamedType('qualifier', univ.Any())
    )


id_ce_certificatePolicies = univ.ObjectIdentifier('2.5.29.32')


class PolicyInformation(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('policyIdentifier', CertPolicyId()),
        namedtype.OptionalNamedType('policyQualifiers', univ.SequenceOf(componentType=PolicyQualifierInfo()).subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, MAX)))
    )


class CertificatePolicies(univ.SequenceOf):
    componentType = PolicyInformation()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


id_ce_policyMappings = univ.ObjectIdentifier('2.5.29.33')


class PolicyMapping(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('issuerDomainPolicy', CertPolicyId()),
        namedtype.NamedType('subjectDomainPolicy', CertPolicyId())
    )


class PolicyMappings(univ.SequenceOf):
    componentType = PolicyMapping()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


id_ce_privateKeyUsagePeriod = univ.ObjectIdentifier('2.5.29.16')


class PrivateKeyUsagePeriod(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('notBefore', useful.GeneralizedTime().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('notAfter', useful.GeneralizedTime().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


id_ce_keyUsage = univ.ObjectIdentifier('2.5.29.15')


class KeyUsage(univ.BitString):
    namedValues = namedval.NamedValues(
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


id_ce = univ.ObjectIdentifier('2.5.29')

id_ce_authorityKeyIdentifier = univ.ObjectIdentifier('2.5.29.35')


class KeyIdentifier(univ.OctetString):
    pass


id_ce_subjectKeyIdentifier = univ.ObjectIdentifier('2.5.29.14')


class SubjectKeyIdentifier(KeyIdentifier):
    pass


id_ce_certificateIssuer = univ.ObjectIdentifier('2.5.29.29')


id_ce_subjectAltName = univ.ObjectIdentifier('2.5.29.17')


id_ce_issuerAltName = univ.ObjectIdentifier('2.5.29.18')


class AttributeValue(univ.Any):
    pass


class AttributeType(univ.ObjectIdentifier):
    pass

certificateAttributesMap = {}


class AttributeTypeAndValue(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', AttributeType()),
        namedtype.NamedType('value', AttributeValue(),
                            openType=opentype.OpenType('type', certificateAttributesMap))
    )


class Attribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', AttributeType()),
        namedtype.NamedType('vals', univ.SetOf(componentType=AttributeValue()))
    )


class SubjectDirectoryAttributes(univ.SequenceOf):
    componentType = Attribute()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class RelativeDistinguishedName(univ.SetOf):
    componentType = AttributeTypeAndValue()


class RDNSequence(univ.SequenceOf):
    componentType = RelativeDistinguishedName()


class Name(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('', RDNSequence())
    )

class CertificateSerialNumber(univ.Integer):
    pass


class AnotherName(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type-id', univ.ObjectIdentifier()),
        namedtype.NamedType('value',
                            univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class GeneralName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('otherName',
                            AnotherName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('rfc822Name',
                            char.IA5String().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('dNSName',
                            char.IA5String().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.NamedType('x400Address',
                            ORAddress().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
        namedtype.NamedType('directoryName',
                            Name().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))),
        namedtype.NamedType('ediPartyName',
                            EDIPartyName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5))),
        namedtype.NamedType('uniformResourceIdentifier',
                            char.IA5String().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 6))),
        namedtype.NamedType('iPAddress', univ.OctetString().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 7))),
        namedtype.NamedType('registeredID', univ.ObjectIdentifier().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 8)))
    )


class GeneralNames(univ.SequenceOf):
    componentType = GeneralName()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class AccessDescription(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('accessMethod', univ.ObjectIdentifier()),
        namedtype.NamedType('accessLocation', GeneralName())
    )


class AuthorityInfoAccessSyntax(univ.SequenceOf):
    componentType = AccessDescription()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class AuthorityKeyIdentifier(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('keyIdentifier', KeyIdentifier().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('authorityCertIssuer', GeneralNames().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('authorityCertSerialNumber', CertificateSerialNumber().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class DistributionPointName(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('fullName', GeneralNames().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.NamedType('nameRelativeToCRLIssuer', RelativeDistinguishedName().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class DistributionPoint(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('distributionPoint', DistributionPointName().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('reasons', ReasonFlags().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('cRLIssuer', GeneralNames().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
    )


class CRLDistPointsSyntax(univ.SequenceOf):
    componentType = DistributionPoint()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class IssuingDistributionPoint(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('distributionPoint', DistributionPointName().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.NamedType('onlyContainsUserCerts', univ.Boolean(False).subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('onlyContainsCACerts', univ.Boolean(False).subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.OptionalNamedType('onlySomeReasons', ReasonFlags().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
        namedtype.NamedType('indirectCRL', univ.Boolean(False).subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4)))
    )


class GeneralSubtree(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('base', GeneralName()),
        namedtype.DefaultedNamedType('minimum', BaseDistance(0).subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('maximum', BaseDistance().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class GeneralSubtrees(univ.SequenceOf):
    componentType = GeneralSubtree()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class NameConstraints(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('permittedSubtrees', GeneralSubtrees().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('excludedSubtrees', GeneralSubtrees().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class CertificateIssuer(GeneralNames):
    pass


class SubjectAltName(GeneralNames):
    pass


class IssuerAltName(GeneralNames):
    pass


certificateExtensionsMap = {}


class Extension(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('extnID', univ.ObjectIdentifier()),
        namedtype.DefaultedNamedType('critical', univ.Boolean('False')),
        namedtype.NamedType('extnValue', univ.OctetString(),
                            openType=opentype.OpenType('extnID', certificateExtensionsMap))
    )


class Extensions(univ.SequenceOf):
    componentType = Extension()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class SubjectPublicKeyInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('algorithm', AlgorithmIdentifier()),
        namedtype.NamedType('subjectPublicKey', univ.BitString())
    )


class UniqueIdentifier(univ.BitString):
    pass


class Time(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('utcTime', useful.UTCTime()),
        namedtype.NamedType('generalTime', useful.GeneralizedTime())
    )


class Validity(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('notBefore', Time()),
        namedtype.NamedType('notAfter', Time())
    )


class Version(univ.Integer):
    namedValues = namedval.NamedValues(
        ('v1', 0), ('v2', 1), ('v3', 2)
    )


class TBSCertificate(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('version', Version('v1').subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
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
        namedtype.OptionalNamedType('extensions', Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
    )


class Certificate(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsCertificate', TBSCertificate()),
        namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('signatureValue', univ.BitString())
    )

# CRL structures

class RevokedCertificate(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('userCertificate', CertificateSerialNumber()),
        namedtype.NamedType('revocationDate', Time()),
        namedtype.OptionalNamedType('crlEntryExtensions', Extensions())
    )


class TBSCertList(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('version', Version()),
        namedtype.NamedType('signature', AlgorithmIdentifier()),
        namedtype.NamedType('issuer', Name()),
        namedtype.NamedType('thisUpdate', Time()),
        namedtype.OptionalNamedType('nextUpdate', Time()),
        namedtype.OptionalNamedType('revokedCertificates', univ.SequenceOf(componentType=RevokedCertificate())),
        namedtype.OptionalNamedType('crlExtensions', Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
    )


class CertificateList(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsCertList', TBSCertList()),
        namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('signature', univ.BitString())
    )

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
    emailAddress: Pkcs9email(),
}

certificateAttributesMap.update(_certificateAttributesMapUpdate)


# map of Certificate Extension OIDs to Extensions

_certificateExtensionsMapUpdate = {
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
    id_ce_cRLDistributionPoints: CRLDistPointsSyntax(),
    id_pe_authorityInfoAccess: AuthorityInfoAccessSyntax(),
    id_ce_cRLNumber: univ.Integer(),
    id_ce_deltaCRLIndicator: BaseCRLNumber(),
    id_ce_issuingDistributionPoint: IssuingDistributionPoint(),
    id_ce_cRLReasons: CRLReason(),
    id_ce_holdInstructionCode: univ.ObjectIdentifier(),
    id_ce_invalidityDate: useful.GeneralizedTime(),
    id_ce_certificateIssuer: GeneralNames(),
}

certificateExtensionsMap.update(_certificateExtensionsMapUpdate)

