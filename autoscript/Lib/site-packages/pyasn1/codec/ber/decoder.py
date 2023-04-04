#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful

__all__ = ['decode']

LOG = debug.registerLoggee(__name__, flags=debug.DEBUG_DECODER)

noValue = base.noValue


class AbstractDecoder(object):
    protoComponent = None

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        raise error.PyAsn1Error('Decoder not implemented for %s' % (tagSet,))

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):
        raise error.PyAsn1Error('Indefinite length mode decoder not implemented for %s' % (tagSet,))


class AbstractSimpleDecoder(AbstractDecoder):
    @staticmethod
    def substrateCollector(asn1Object, substrate, length):
        return substrate[:length], substrate[length:]

    def _createComponent(self, asn1Spec, tagSet, value, **options):
        if options.get('native'):
            return value
        elif asn1Spec is None:
            return self.protoComponent.clone(value, tagSet=tagSet)
        elif value is noValue:
            return asn1Spec
        else:
            return asn1Spec.clone(value)


class ExplicitTagDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Any('')

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        if substrateFun:
            return substrateFun(
                self._createComponent(asn1Spec, tagSet, '', **options),
                substrate, length
            )

        head, tail = substrate[:length], substrate[length:]

        value, _ = decodeFun(head, asn1Spec, tagSet, length, **options)

        if LOG:
            LOG('explicit tag container carries %d octets of trailing payload '
                '(will be lost!): %s' % (len(_), debug.hexdump(_)))

        return value, tail

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):
        if substrateFun:
            return substrateFun(
                self._createComponent(asn1Spec, tagSet, '', **options),
                substrate, length
            )

        value, substrate = decodeFun(substrate, asn1Spec, tagSet, length, **options)

        eooMarker, substrate = decodeFun(substrate, allowEoo=True, **options)

        if eooMarker is eoo.endOfOctets:
            return value, substrate
        else:
            raise error.PyAsn1Error('Missing end-of-octets terminator')


explicitTagDecoder = ExplicitTagDecoder()


class IntegerDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Integer(0)

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):

        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')

        head, tail = substrate[:length], substrate[length:]

        if not head:
            return self._createComponent(asn1Spec, tagSet, 0, **options), tail

        value = from_bytes(head, signed=True)

        return self._createComponent(asn1Spec, tagSet, value, **options), tail


class BooleanDecoder(IntegerDecoder):
    protoComponent = univ.Boolean(0)

    def _createComponent(self, asn1Spec, tagSet, value, **options):
        return IntegerDecoder._createComponent(
            self, asn1Spec, tagSet, value and 1 or 0, **options)


class BitStringDecoder(AbstractSimpleDecoder):
    protoComponent = univ.BitString(())
    supportConstructedForm = True

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        head, tail = substrate[:length], substrate[length:]

        if substrateFun:
            return substrateFun(self._createComponent(
                asn1Spec, tagSet, noValue, **options), substrate, length)

        if not head:
            raise error.PyAsn1Error('Empty BIT STRING substrate')

        if tagSet[0].tagFormat == tag.tagFormatSimple:  # XXX what tag to check?

            trailingBits = oct2int(head[0])
            if trailingBits > 7:
                raise error.PyAsn1Error(
                    'Trailing bits overflow %s' % trailingBits
                )

            value = self.protoComponent.fromOctetString(
                head[1:], internalFormat=True, padding=trailingBits)

            return self._createComponent(asn1Spec, tagSet, value, **options), tail

        if not self.supportConstructedForm:
            raise error.PyAsn1Error('Constructed encoding form prohibited '
                                    'at %s' % self.__class__.__name__)

        if LOG:
            LOG('assembling constructed serialization')

        # All inner fragments are of the same type, treat them as octet string
        substrateFun = self.substrateCollector

        bitString = self.protoComponent.fromOctetString(null, internalFormat=True)

        while head:
            component, head = decodeFun(head, self.protoComponent,
                                        substrateFun=substrateFun, **options)

            trailingBits = oct2int(component[0])
            if trailingBits > 7:
                raise error.PyAsn1Error(
                    'Trailing bits overflow %s' % trailingBits
                )

            bitString = self.protoComponent.fromOctetString(
                component[1:], internalFormat=True,
                prepend=bitString, padding=trailingBits
            )

        return self._createComponent(asn1Spec, tagSet, bitString, **options), tail

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):

        if substrateFun:
            return substrateFun(self._createComponent(asn1Spec, tagSet, noValue, **options), substrate, length)

        # All inner fragments are of the same type, treat them as octet string
        substrateFun = self.substrateCollector

        bitString = self.protoComponent.fromOctetString(null, internalFormat=True)

        while substrate:
            component, substrate = decodeFun(substrate, self.protoComponent,
                                             substrateFun=substrateFun,
                                             allowEoo=True, **options)
            if component is eoo.endOfOctets:
                break

            trailingBits = oct2int(component[0])
            if trailingBits > 7:
                raise error.PyAsn1Error(
                    'Trailing bits overflow %s' % trailingBits
                )

            bitString = self.protoComponent.fromOctetString(
                component[1:], internalFormat=True,
                prepend=bitString, padding=trailingBits
            )

        else:
            raise error.SubstrateUnderrunError('No EOO seen before substrate ends')

        return self._createComponent(asn1Spec, tagSet, bitString, **options), substrate


class OctetStringDecoder(AbstractSimpleDecoder):
    protoComponent = univ.OctetString('')
    supportConstructedForm = True

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        head, tail = substrate[:length], substrate[length:]

        if substrateFun:
            return substrateFun(self._createComponent(asn1Spec, tagSet, noValue, **options),
                                substrate, length)

        if tagSet[0].tagFormat == tag.tagFormatSimple:  # XXX what tag to check?
            return self._createComponent(asn1Spec, tagSet, head, **options), tail

        if not self.supportConstructedForm:
            raise error.PyAsn1Error('Constructed encoding form prohibited at %s' % self.__class__.__name__)

        if LOG:
            LOG('assembling constructed serialization')

        # All inner fragments are of the same type, treat them as octet string
        substrateFun = self.substrateCollector

        header = null

        while head:
            component, head = decodeFun(head, self.protoComponent,
                                        substrateFun=substrateFun,
                                        **options)
            header += component

        return self._createComponent(asn1Spec, tagSet, header, **options), tail

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):
        if substrateFun and substrateFun is not self.substrateCollector:
            asn1Object = self._createComponent(asn1Spec, tagSet, noValue, **options)
            return substrateFun(asn1Object, substrate, length)

        # All inner fragments are of the same type, treat them as octet string
        substrateFun = self.substrateCollector

        header = null

        while substrate:
            component, substrate = decodeFun(substrate,
                                             self.protoComponent,
                                             substrateFun=substrateFun,
                                             allowEoo=True, **options)
            if component is eoo.endOfOctets:
                break

            header += component

        else:
            raise error.SubstrateUnderrunError(
                'No EOO seen before substrate ends'
            )

        return self._createComponent(asn1Spec, tagSet, header, **options), substrate


class NullDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Null('')

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):

        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')

        head, tail = substrate[:length], substrate[length:]

        component = self._createComponent(asn1Spec, tagSet, '', **options)

        if head:
            raise error.PyAsn1Error('Unexpected %d-octet substrate for Null' % length)

        return component, tail


class ObjectIdentifierDecoder(AbstractSimpleDecoder):
    protoComponent = univ.ObjectIdentifier(())

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')

        head, tail = substrate[:length], substrate[length:]
        if not head:
            raise error.PyAsn1Error('Empty substrate')

        head = octs2ints(head)

        oid = ()
        index = 0
        substrateLen = len(head)
        while index < substrateLen:
            subId = head[index]
            index += 1
            if subId < 128:
                oid += (subId,)
            elif subId > 128:
                # Construct subid from a number of octets
                nextSubId = subId
                subId = 0
                while nextSubId >= 128:
                    subId = (subId << 7) + (nextSubId & 0x7F)
                    if index >= substrateLen:
                        raise error.SubstrateUnderrunError(
                            'Short substrate for sub-OID past %s' % (oid,)
                        )
                    nextSubId = head[index]
                    index += 1
                oid += ((subId << 7) + nextSubId,)
            elif subId == 128:
                # ASN.1 spec forbids leading zeros (0x80) in OID
                # encoding, tolerating it opens a vulnerability. See
                # https://www.esat.kuleuven.be/cosic/publications/article-1432.pdf
                # page 7
                raise error.PyAsn1Error('Invalid octet 0x80 in OID encoding')

        # Decode two leading arcs
        if 0 <= oid[0] <= 39:
            oid = (0,) + oid
        elif 40 <= oid[0] <= 79:
            oid = (1, oid[0] - 40) + oid[1:]
        elif oid[0] >= 80:
            oid = (2, oid[0] - 80) + oid[1:]
        else:
            raise error.PyAsn1Error('Malformed first OID octet: %s' % head[0])

        return self._createComponent(asn1Spec, tagSet, oid, **options), tail


class RealDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Real()

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')

        head, tail = substrate[:length], substrate[length:]

        if not head:
            return self._createComponent(asn1Spec, tagSet, 0.0, **options), tail

        fo = oct2int(head[0])
        head = head[1:]
        if fo & 0x80:  # binary encoding
            if not head:
                raise error.PyAsn1Error("Incomplete floating-point value")

            if LOG:
                LOG('decoding binary encoded REAL')

            n = (fo & 0x03) + 1

            if n == 4:
                n = oct2int(head[0])
                head = head[1:]

            eo, head = head[:n], head[n:]

            if not eo or not head:
                raise error.PyAsn1Error('Real exponent screwed')

            e = oct2int(eo[0]) & 0x80 and -1 or 0

            while eo:  # exponent
                e <<= 8
                e |= oct2int(eo[0])
                eo = eo[1:]

            b = fo >> 4 & 0x03  # base bits

            if b > 2:
                raise error.PyAsn1Error('Illegal Real base')

            if b == 1:  # encbase = 8
                e *= 3

            elif b == 2:  # encbase = 16
                e *= 4
            p = 0

            while head:  # value
                p <<= 8
                p |= oct2int(head[0])
                head = head[1:]

            if fo & 0x40:  # sign bit
                p = -p

            sf = fo >> 2 & 0x03  # scale bits
            p *= 2 ** sf
            value = (p, 2, e)

        elif fo & 0x40:  # infinite value
            if LOG:
                LOG('decoding infinite REAL')

            value = fo & 0x01 and '-inf' or 'inf'

        elif fo & 0xc0 == 0:  # character encoding
            if not head:
                raise error.PyAsn1Error("Incomplete floating-point value")

            if LOG:
                LOG('decoding character encoded REAL')

            try:
                if fo & 0x3 == 0x1:  # NR1
                    value = (int(head), 10, 0)

                elif fo & 0x3 == 0x2:  # NR2
                    value = float(head)

                elif fo & 0x3 == 0x3:  # NR3
                    value = float(head)

                else:
                    raise error.SubstrateUnderrunError(
                        'Unknown NR (tag %s)' % fo
                    )

            except ValueError:
                raise error.SubstrateUnderrunError(
                    'Bad character Real syntax'
                )

        else:
            raise error.SubstrateUnderrunError(
                'Unknown encoding (tag %s)' % fo
            )

        return self._createComponent(asn1Spec, tagSet, value, **options), tail


class AbstractConstructedDecoder(AbstractDecoder):
    protoComponent = None


class UniversalConstructedTypeDecoder(AbstractConstructedDecoder):
    protoRecordComponent = None
    protoSequenceComponent = None

    def _getComponentTagMap(self, asn1Object, idx):
        raise NotImplementedError()

    def _getComponentPositionByType(self, asn1Object, tagSet, idx):
        raise NotImplementedError()

    def _decodeComponents(self, substrate, tagSet=None, decodeFun=None, **options):
        components = []
        componentTypes = set()

        while substrate:
            component, substrate = decodeFun(substrate, **options)
            if component is eoo.endOfOctets:
                break

            components.append(component)
            componentTypes.add(component.tagSet)

        # Now we have to guess is it SEQUENCE/SET or SEQUENCE OF/SET OF
        # The heuristics is:
        # * 1+ components of different types -> likely SEQUENCE/SET
        # * otherwise -> likely SEQUENCE OF/SET OF
        if len(componentTypes) > 1:
            protoComponent = self.protoRecordComponent

        else:
            protoComponent = self.protoSequenceComponent

        asn1Object = protoComponent.clone(
            # construct tagSet from base tag from prototype ASN.1 object
            # and additional tags recovered from the substrate
            tagSet=tag.TagSet(protoComponent.tagSet.baseTag, *tagSet.superTags)
        )

        if LOG:
            LOG('guessed %r container type (pass `asn1Spec` to guide the '
                'decoder)' % asn1Object)

        for idx, component in enumerate(components):
            asn1Object.setComponentByPosition(
                idx, component,
                verifyConstraints=False,
                matchTags=False, matchConstraints=False
            )

        return asn1Object, substrate

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        if tagSet[0].tagFormat != tag.tagFormatConstructed:
            raise error.PyAsn1Error('Constructed tag format expected')

        head, tail = substrate[:length], substrate[length:]

        if substrateFun is not None:
            if asn1Spec is not None:
                asn1Object = asn1Spec.clone()

            elif self.protoComponent is not None:
                asn1Object = self.protoComponent.clone(tagSet=tagSet)

            else:
                asn1Object = self.protoRecordComponent, self.protoSequenceComponent

            return substrateFun(asn1Object, substrate, length)

        if asn1Spec is None:
            asn1Object, trailing = self._decodeComponents(
                head, tagSet=tagSet, decodeFun=decodeFun, **options
            )

            if trailing:
                if LOG:
                    LOG('Unused trailing %d octets encountered: %s' % (
                        len(trailing), debug.hexdump(trailing)))

            return asn1Object, tail

        asn1Object = asn1Spec.clone()
        asn1Object.clear()

        if asn1Spec.typeId in (univ.Sequence.typeId, univ.Set.typeId):

            namedTypes = asn1Spec.componentType

            isSetType = asn1Spec.typeId == univ.Set.typeId
            isDeterministic = not isSetType and not namedTypes.hasOptionalOrDefault

            if LOG:
                LOG('decoding %sdeterministic %s type %r chosen by type ID' % (
                    not isDeterministic and 'non-' or '', isSetType and 'SET' or '',
                    asn1Spec))

            seenIndices = set()
            idx = 0
            while head:
                if not namedTypes:
                    componentType = None

                elif isSetType:
                    componentType = namedTypes.tagMapUnique

                else:
                    try:
                        if isDeterministic:
                            componentType = namedTypes[idx].asn1Object

                        elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                            componentType = namedTypes.getTagMapNearPosition(idx)

                        else:
                            componentType = namedTypes[idx].asn1Object

                    except IndexError:
                        raise error.PyAsn1Error(
                            'Excessive components decoded at %r' % (asn1Spec,)
                        )

                component, head = decodeFun(head, componentType, **options)

                if not isDeterministic and namedTypes:
                    if isSetType:
                        idx = namedTypes.getPositionByType(component.effectiveTagSet)

                    elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                        idx = namedTypes.getPositionNearType(component.effectiveTagSet, idx)

                asn1Object.setComponentByPosition(
                    idx, component,
                    verifyConstraints=False,
                    matchTags=False, matchConstraints=False
                )

                seenIndices.add(idx)
                idx += 1

            if LOG:
                LOG('seen component indices %s' % seenIndices)

            if namedTypes:
                if not namedTypes.requiredComponents.issubset(seenIndices):
                    raise error.PyAsn1Error(
                        'ASN.1 object %s has uninitialized '
                        'components' % asn1Object.__class__.__name__)

                if  namedTypes.hasOpenTypes:

                    openTypes = options.get('openTypes', {})

                    if LOG:
                        LOG('user-specified open types map:')

                        for k, v in openTypes.items():
                            LOG('%s -> %r' % (k, v))

                    if openTypes or options.get('decodeOpenTypes', False):

                        for idx, namedType in enumerate(namedTypes.namedTypes):
                            if not namedType.openType:
                                continue

                            if namedType.isOptional and not asn1Object.getComponentByPosition(idx).isValue:
                                continue

                            governingValue = asn1Object.getComponentByName(
                                namedType.openType.name
                            )

                            try:
                                openType = openTypes[governingValue]

                            except KeyError:

                                if LOG:
                                    LOG('default open types map of component '
                                        '"%s.%s" governed by component "%s.%s"'
                                        ':' % (asn1Object.__class__.__name__,
                                               namedType.name,
                                               asn1Object.__class__.__name__,
                                               namedType.openType.name))

                                    for k, v in namedType.openType.items():
                                        LOG('%s -> %r' % (k, v))

                                try:
                                    openType = namedType.openType[governingValue]

                                except KeyError:
                                    if LOG:
                                        LOG('failed to resolve open type by governing '
                                            'value %r' % (governingValue,))
                                    continue

                            if LOG:
                                LOG('resolved open type %r by governing '
                                    'value %r' % (openType, governingValue))

                            containerValue = asn1Object.getComponentByPosition(idx)

                            if containerValue.typeId in (
                                    univ.SetOf.typeId, univ.SequenceOf.typeId):

                                for pos, containerElement in enumerate(
                                        containerValue):

                                    component, rest = decodeFun(
                                        containerValue[pos].asOctets(),
                                        asn1Spec=openType, **options
                                    )

                                    containerValue[pos] = component

                            else:
                                component, rest = decodeFun(
                                    asn1Object.getComponentByPosition(idx).asOctets(),
                                    asn1Spec=openType, **options
                                )

                                asn1Object.setComponentByPosition(idx, component)

            else:
                inconsistency = asn1Object.isInconsistent
                if inconsistency:
                    raise inconsistency

        else:
            asn1Object = asn1Spec.clone()
            asn1Object.clear()

            componentType = asn1Spec.componentType

            if LOG:
                LOG('decoding type %r chosen by given `asn1Spec`' % componentType)

            idx = 0

            while head:
                component, head = decodeFun(head, componentType, **options)
                asn1Object.setComponentByPosition(
                    idx, component,
                    verifyConstraints=False,
                    matchTags=False, matchConstraints=False
                )

                idx += 1

        return asn1Object, tail

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):
        if tagSet[0].tagFormat != tag.tagFormatConstructed:
            raise error.PyAsn1Error('Constructed tag format expected')

        if substrateFun is not None:
            if asn1Spec is not None:
                asn1Object = asn1Spec.clone()

            elif self.protoComponent is not None:
                asn1Object = self.protoComponent.clone(tagSet=tagSet)

            else:
                asn1Object = self.protoRecordComponent, self.protoSequenceComponent

            return substrateFun(asn1Object, substrate, length)

        if asn1Spec is None:
            return self._decodeComponents(
                substrate, tagSet=tagSet, decodeFun=decodeFun,
                **dict(options, allowEoo=True)
            )

        asn1Object = asn1Spec.clone()
        asn1Object.clear()

        if asn1Spec.typeId in (univ.Sequence.typeId, univ.Set.typeId):

            namedTypes = asn1Object.componentType

            isSetType = asn1Object.typeId == univ.Set.typeId
            isDeterministic = not isSetType and not namedTypes.hasOptionalOrDefault

            if LOG:
                LOG('decoding %sdeterministic %s type %r chosen by type ID' % (
                    not isDeterministic and 'non-' or '', isSetType and 'SET' or '',
                    asn1Spec))

            seenIndices = set()
            idx = 0
            while substrate:
                if len(namedTypes) <= idx:
                    asn1Spec = None

                elif isSetType:
                    asn1Spec = namedTypes.tagMapUnique

                else:
                    try:
                        if isDeterministic:
                            asn1Spec = namedTypes[idx].asn1Object

                        elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                            asn1Spec = namedTypes.getTagMapNearPosition(idx)

                        else:
                            asn1Spec = namedTypes[idx].asn1Object

                    except IndexError:
                        raise error.PyAsn1Error(
                            'Excessive components decoded at %r' % (asn1Object,)
                        )

                component, substrate = decodeFun(substrate, asn1Spec, allowEoo=True, **options)
                if component is eoo.endOfOctets:
                    break

                if not isDeterministic and namedTypes:
                    if isSetType:
                        idx = namedTypes.getPositionByType(component.effectiveTagSet)
                    elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                        idx = namedTypes.getPositionNearType(component.effectiveTagSet, idx)

                asn1Object.setComponentByPosition(
                    idx, component,
                    verifyConstraints=False,
                    matchTags=False, matchConstraints=False
                )

                seenIndices.add(idx)
                idx += 1

            else:
                raise error.SubstrateUnderrunError(
                    'No EOO seen before substrate ends'
                )

            if LOG:
                LOG('seen component indices %s' % seenIndices)

            if namedTypes:
                if not namedTypes.requiredComponents.issubset(seenIndices):
                    raise error.PyAsn1Error('ASN.1 object %s has uninitialized components' % asn1Object.__class__.__name__)

                if namedTypes.hasOpenTypes:

                    openTypes = options.get('openTypes', {})

                    if LOG:
                        LOG('user-specified open types map:')

                        for k, v in openTypes.items():
                            LOG('%s -> %r' % (k, v))

                    if openTypes or options.get('decodeOpenTypes', False):

                        for idx, namedType in enumerate(namedTypes.namedTypes):
                            if not namedType.openType:
                                continue

                            if namedType.isOptional and not asn1Object.getComponentByPosition(idx).isValue:
                                continue

                            governingValue = asn1Object.getComponentByName(
                                namedType.openType.name
                            )

                            try:
                                openType = openTypes[governingValue]

                            except KeyError:

                                if LOG:
                                    LOG('default open types map of component '
                                        '"%s.%s" governed by component "%s.%s"'
                                        ':' % (asn1Object.__class__.__name__,
                                               namedType.name,
                                               asn1Object.__class__.__name__,
                                               namedType.openType.name))

                                    for k, v in namedType.openType.items():
                                        LOG('%s -> %r' % (k, v))

                                try:
                                    openType = namedType.openType[governingValue]

                                except KeyError:
                                    if LOG:
                                        LOG('failed to resolve open type by governing '
                                            'value %r' % (governingValue,))
                                    continue

                            if LOG:
                                LOG('resolved open type %r by governing '
                                    'value %r' % (openType, governingValue))

                            containerValue = asn1Object.getComponentByPosition(idx)

                            if containerValue.typeId in (
                                    univ.SetOf.typeId, univ.SequenceOf.typeId):

                                for pos, containerElement in enumerate(
                                        containerValue):

                                    component, rest = decodeFun(
                                        containerValue[pos].asOctets(),
                                        asn1Spec=openType, **dict(options, allowEoo=True)
                                    )

                                    containerValue[pos] = component

                            else:
                                component, rest = decodeFun(
                                    asn1Object.getComponentByPosition(idx).asOctets(),
                                    asn1Spec=openType, **dict(options, allowEoo=True)
                                )

                                if component is not eoo.endOfOctets:
                                    asn1Object.setComponentByPosition(idx, component)

                else:
                    inconsistency = asn1Object.isInconsistent
                    if inconsistency:
                        raise inconsistency

        else:
            asn1Object = asn1Spec.clone()
            asn1Object.clear()

            componentType = asn1Spec.componentType

            if LOG:
                LOG('decoding type %r chosen by given `asn1Spec`' % componentType)

            idx = 0

            while substrate:
                component, substrate = decodeFun(substrate, componentType, allowEoo=True, **options)

                if component is eoo.endOfOctets:
                    break

                asn1Object.setComponentByPosition(
                    idx, component,
                    verifyConstraints=False,
                    matchTags=False, matchConstraints=False
                )

                idx += 1

            else:
                raise error.SubstrateUnderrunError(
                    'No EOO seen before substrate ends'
                )

        return asn1Object, substrate


class SequenceOrSequenceOfDecoder(UniversalConstructedTypeDecoder):
    protoRecordComponent = univ.Sequence()
    protoSequenceComponent = univ.SequenceOf()


class SequenceDecoder(SequenceOrSequenceOfDecoder):
    protoComponent = univ.Sequence()


class SequenceOfDecoder(SequenceOrSequenceOfDecoder):
    protoComponent = univ.SequenceOf()


class SetOrSetOfDecoder(UniversalConstructedTypeDecoder):
    protoRecordComponent = univ.Set()
    protoSequenceComponent = univ.SetOf()


class SetDecoder(SetOrSetOfDecoder):
    protoComponent = univ.Set()



class SetOfDecoder(SetOrSetOfDecoder):
    protoComponent = univ.SetOf()


class ChoiceDecoder(AbstractConstructedDecoder):
    protoComponent = univ.Choice()

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        head, tail = substrate[:length], substrate[length:]

        if asn1Spec is None:
            asn1Object = self.protoComponent.clone(tagSet=tagSet)

        else:
            asn1Object = asn1Spec.clone()

        if substrateFun:
            return substrateFun(asn1Object, substrate, length)

        if asn1Object.tagSet == tagSet:
            if LOG:
                LOG('decoding %s as explicitly tagged CHOICE' % (tagSet,))

            component, head = decodeFun(
                head, asn1Object.componentTagMap, **options
            )

        else:
            if LOG:
                LOG('decoding %s as untagged CHOICE' % (tagSet,))

            component, head = decodeFun(
                head, asn1Object.componentTagMap,
                tagSet, length, state, **options
            )

        effectiveTagSet = component.effectiveTagSet

        if LOG:
            LOG('decoded component %s, effective tag set %s' % (component, effectiveTagSet))

        asn1Object.setComponentByType(
            effectiveTagSet, component,
            verifyConstraints=False,
            matchTags=False, matchConstraints=False,
            innerFlag=False
        )

        return asn1Object, tail

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):
        if asn1Spec is None:
            asn1Object = self.protoComponent.clone(tagSet=tagSet)
        else:
            asn1Object = asn1Spec.clone()

        if substrateFun:
            return substrateFun(asn1Object, substrate, length)

        if asn1Object.tagSet == tagSet:
            if LOG:
                LOG('decoding %s as explicitly tagged CHOICE' % (tagSet,))

            component, substrate = decodeFun(
                substrate, asn1Object.componentType.tagMapUnique, **options
            )

            # eat up EOO marker
            eooMarker, substrate = decodeFun(
                substrate, allowEoo=True, **options
            )

            if eooMarker is not eoo.endOfOctets:
                raise error.PyAsn1Error('No EOO seen before substrate ends')

        else:
            if LOG:
                LOG('decoding %s as untagged CHOICE' % (tagSet,))

            component, substrate = decodeFun(
                substrate, asn1Object.componentType.tagMapUnique,
                tagSet, length, state, **options
            )

        effectiveTagSet = component.effectiveTagSet

        if LOG:
            LOG('decoded component %s, effective tag set %s' % (component, effectiveTagSet))

        asn1Object.setComponentByType(
            effectiveTagSet, component,
            verifyConstraints=False,
            matchTags=False, matchConstraints=False,
            innerFlag=False
        )

        return asn1Object, substrate


class AnyDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Any()

    def valueDecoder(self, substrate, asn1Spec,
                     tagSet=None, length=None, state=None,
                     decodeFun=None, substrateFun=None,
                     **options):
        if asn1Spec is None:
            isUntagged = True

        elif asn1Spec.__class__ is tagmap.TagMap:
            isUntagged = tagSet not in asn1Spec.tagMap

        else:
            isUntagged = tagSet != asn1Spec.tagSet

        if isUntagged:
            fullSubstrate = options['fullSubstrate']

            # untagged Any container, recover inner header substrate
            length += len(fullSubstrate) - len(substrate)
            substrate = fullSubstrate

            if LOG:
                LOG('decoding as untagged ANY, substrate %s' % debug.hexdump(substrate))

        if substrateFun:
            return substrateFun(self._createComponent(asn1Spec, tagSet, noValue, **options),
                                substrate, length)

        head, tail = substrate[:length], substrate[length:]

        return self._createComponent(asn1Spec, tagSet, head, **options), tail

    def indefLenValueDecoder(self, substrate, asn1Spec,
                             tagSet=None, length=None, state=None,
                             decodeFun=None, substrateFun=None,
                             **options):
        if asn1Spec is None:
            isTagged = False

        elif asn1Spec.__class__ is tagmap.TagMap:
            isTagged = tagSet in asn1Spec.tagMap

        else:
            isTagged = tagSet == asn1Spec.tagSet

        if isTagged:
            # tagged Any type -- consume header substrate
            header = null

            if LOG:
                LOG('decoding as tagged ANY')

        else:
            fullSubstrate = options['fullSubstrate']

            # untagged Any, recover header substrate
            header = fullSubstrate[:-len(substrate)]

            if LOG:
                LOG('decoding as untagged ANY, header substrate %s' % debug.hexdump(header))

        # Any components do not inherit initial tag
        asn1Spec = self.protoComponent

        if substrateFun and substrateFun is not self.substrateCollector:
            asn1Object = self._createComponent(asn1Spec, tagSet, noValue, **options)
            return substrateFun(asn1Object, header + substrate, length + len(header))

        if LOG:
            LOG('assembling constructed serialization')

        # All inner fragments are of the same type, treat them as octet string
        substrateFun = self.substrateCollector

        while substrate:
            component, substrate = decodeFun(substrate, asn1Spec,
                                             substrateFun=substrateFun,
                                             allowEoo=True, **options)
            if component is eoo.endOfOctets:
                break

            header += component

        else:
            raise error.SubstrateUnderrunError(
                'No EOO seen before substrate ends'
            )

        if substrateFun:
            return header, substrate

        else:
            return self._createComponent(asn1Spec, tagSet, header, **options), substrate


# character string types
class UTF8StringDecoder(OctetStringDecoder):
    protoComponent = char.UTF8String()


class NumericStringDecoder(OctetStringDecoder):
    protoComponent = char.NumericString()


class PrintableStringDecoder(OctetStringDecoder):
    protoComponent = char.PrintableString()


class TeletexStringDecoder(OctetStringDecoder):
    protoComponent = char.TeletexString()


class VideotexStringDecoder(OctetStringDecoder):
    protoComponent = char.VideotexString()


class IA5StringDecoder(OctetStringDecoder):
    protoComponent = char.IA5String()


class GraphicStringDecoder(OctetStringDecoder):
    protoComponent = char.GraphicString()


class VisibleStringDecoder(OctetStringDecoder):
    protoComponent = char.VisibleString()


class GeneralStringDecoder(OctetStringDecoder):
    protoComponent = char.GeneralString()


class UniversalStringDecoder(OctetStringDecoder):
    protoComponent = char.UniversalString()


class BMPStringDecoder(OctetStringDecoder):
    protoComponent = char.BMPString()


# "useful" types
class ObjectDescriptorDecoder(OctetStringDecoder):
    protoComponent = useful.ObjectDescriptor()


class GeneralizedTimeDecoder(OctetStringDecoder):
    protoComponent = useful.GeneralizedTime()


class UTCTimeDecoder(OctetStringDecoder):
    protoComponent = useful.UTCTime()


tagMap = {
    univ.Integer.tagSet: IntegerDecoder(),
    univ.Boolean.tagSet: BooleanDecoder(),
    univ.BitString.tagSet: BitStringDecoder(),
    univ.OctetString.tagSet: OctetStringDecoder(),
    univ.Null.tagSet: NullDecoder(),
    univ.ObjectIdentifier.tagSet: ObjectIdentifierDecoder(),
    univ.Enumerated.tagSet: IntegerDecoder(),
    univ.Real.tagSet: RealDecoder(),
    univ.Sequence.tagSet: SequenceOrSequenceOfDecoder(),  # conflicts with SequenceOf
    univ.Set.tagSet: SetOrSetOfDecoder(),  # conflicts with SetOf
    univ.Choice.tagSet: ChoiceDecoder(),  # conflicts with Any
    # character string types
    char.UTF8String.tagSet: UTF8StringDecoder(),
    char.NumericString.tagSet: NumericStringDecoder(),
    char.PrintableString.tagSet: PrintableStringDecoder(),
    char.TeletexString.tagSet: TeletexStringDecoder(),
    char.VideotexString.tagSet: VideotexStringDecoder(),
    char.IA5String.tagSet: IA5StringDecoder(),
    char.GraphicString.tagSet: GraphicStringDecoder(),
    char.VisibleString.tagSet: VisibleStringDecoder(),
    char.GeneralString.tagSet: GeneralStringDecoder(),
    char.UniversalString.tagSet: UniversalStringDecoder(),
    char.BMPString.tagSet: BMPStringDecoder(),
    # useful types
    useful.ObjectDescriptor.tagSet: ObjectDescriptorDecoder(),
    useful.GeneralizedTime.tagSet: GeneralizedTimeDecoder(),
    useful.UTCTime.tagSet: UTCTimeDecoder()
}

# Type-to-codec map for ambiguous ASN.1 types
typeMap = {
    univ.Set.typeId: SetDecoder(),
    univ.SetOf.typeId: SetOfDecoder(),
    univ.Sequence.typeId: SequenceDecoder(),
    univ.SequenceOf.typeId: SequenceOfDecoder(),
    univ.Choice.typeId: ChoiceDecoder(),
    univ.Any.typeId: AnyDecoder()
}

# Put in non-ambiguous types for faster codec lookup
for typeDecoder in tagMap.values():
    if typeDecoder.protoComponent is not None:
        typeId = typeDecoder.protoComponent.__class__.typeId
        if typeId is not None and typeId not in typeMap:
            typeMap[typeId] = typeDecoder


(stDecodeTag,
 stDecodeLength,
 stGetValueDecoder,
 stGetValueDecoderByAsn1Spec,
 stGetValueDecoderByTag,
 stTryAsExplicitTag,
 stDecodeValue,
 stDumpRawValue,
 stErrorCondition,
 stStop) = [x for x in range(10)]


class Decoder(object):
    defaultErrorState = stErrorCondition
    #defaultErrorState = stDumpRawValue
    defaultRawDecoder = AnyDecoder()
    supportIndefLength = True

    # noinspection PyDefaultArgument
    def __init__(self, tagMap, typeMap={}):
        self.__tagMap = tagMap
        self.__typeMap = typeMap
        # Tag & TagSet objects caches
        self.__tagCache = {}
        self.__tagSetCache = {}
        self.__eooSentinel = ints2octs((0, 0))

    def __call__(self, substrate, asn1Spec=None,
                 tagSet=None, length=None, state=stDecodeTag,
                 decodeFun=None, substrateFun=None,
                 **options):

        if LOG:
            LOG('decoder called at scope %s with state %d, working with up to %d octets of substrate: %s' % (debug.scope, state, len(substrate), debug.hexdump(substrate)))

        allowEoo = options.pop('allowEoo', False)

        # Look for end-of-octets sentinel
        if allowEoo and self.supportIndefLength:
            if substrate[:2] == self.__eooSentinel:
                if LOG:
                    LOG('end-of-octets sentinel found')
                return eoo.endOfOctets, substrate[2:]

        value = noValue

        tagMap = self.__tagMap
        typeMap = self.__typeMap
        tagCache = self.__tagCache
        tagSetCache = self.__tagSetCache

        fullSubstrate = substrate

        while state is not stStop:

            if state is stDecodeTag:
                if not substrate:
                    raise error.SubstrateUnderrunError(
                        'Short octet stream on tag decoding'
                    )

                # Decode tag
                isShortTag = True
                firstOctet = substrate[0]
                substrate = substrate[1:]

                try:
                    lastTag = tagCache[firstOctet]

                except KeyError:
                    integerTag = oct2int(firstOctet)
                    tagClass = integerTag & 0xC0
                    tagFormat = integerTag & 0x20
                    tagId = integerTag & 0x1F

                    if tagId == 0x1F:
                        isShortTag = False
                        lengthOctetIdx = 0
                        tagId = 0

                        try:
                            while True:
                                integerTag = oct2int(substrate[lengthOctetIdx])
                                lengthOctetIdx += 1
                                tagId <<= 7
                                tagId |= (integerTag & 0x7F)
                                if not integerTag & 0x80:
                                    break

                            substrate = substrate[lengthOctetIdx:]

                        except IndexError:
                            raise error.SubstrateUnderrunError(
                                'Short octet stream on long tag decoding'
                            )

                    lastTag = tag.Tag(
                        tagClass=tagClass, tagFormat=tagFormat, tagId=tagId
                    )

                    if isShortTag:
                        # cache short tags
                        tagCache[firstOctet] = lastTag

                if tagSet is None:
                    if isShortTag:
                        try:
                            tagSet = tagSetCache[firstOctet]

                        except KeyError:
                            # base tag not recovered
                            tagSet = tag.TagSet((), lastTag)
                            tagSetCache[firstOctet] = tagSet
                    else:
                        tagSet = tag.TagSet((), lastTag)

                else:
                    tagSet = lastTag + tagSet

                state = stDecodeLength

                if LOG:
                    LOG('tag decoded into %s, decoding length' % tagSet)

            if state is stDecodeLength:
                # Decode length
                if not substrate:
                    raise error.SubstrateUnderrunError(
                        'Short octet stream on length decoding'
                    )

                firstOctet = oct2int(substrate[0])

                if firstOctet < 128:
                    size = 1
                    length = firstOctet

                elif firstOctet > 128:
                    size = firstOctet & 0x7F
                    # encoded in size bytes
                    encodedLength = octs2ints(substrate[1:size + 1])
                    # missing check on maximum size, which shouldn't be a
                    # problem, we can handle more than is possible
                    if len(encodedLength) != size:
                        raise error.SubstrateUnderrunError(
                            '%s<%s at %s' % (size, len(encodedLength), tagSet)
                        )

                    length = 0
                    for lengthOctet in encodedLength:
                        length <<= 8
                        length |= lengthOctet
                    size += 1

                else:
                    size = 1
                    length = -1

                substrate = substrate[size:]

                if length == -1:
                    if not self.supportIndefLength:
                        raise error.PyAsn1Error('Indefinite length encoding not supported by this codec')

                else:
                    if len(substrate) < length:
                        raise error.SubstrateUnderrunError('%d-octet short' % (length - len(substrate)))

                state = stGetValueDecoder

                if LOG:
                    LOG('value length decoded into %d, payload substrate is: %s' % (length, debug.hexdump(length == -1 and substrate or substrate[:length])))

            if state is stGetValueDecoder:
                if asn1Spec is None:
                    state = stGetValueDecoderByTag

                else:
                    state = stGetValueDecoderByAsn1Spec
            #
            # There're two ways of creating subtypes in ASN.1 what influences
            # decoder operation. These methods are:
            # 1) Either base types used in or no IMPLICIT tagging has been
            #    applied on subtyping.
            # 2) Subtype syntax drops base type information (by means of
            #    IMPLICIT tagging.
            # The first case allows for complete tag recovery from substrate
            # while the second one requires original ASN.1 type spec for
            # decoding.
            #
            # In either case a set of tags (tagSet) is coming from substrate
            # in an incremental, tag-by-tag fashion (this is the case of
            # EXPLICIT tag which is most basic). Outermost tag comes first
            # from the wire.
            #
            if state is stGetValueDecoderByTag:
                try:
                    concreteDecoder = tagMap[tagSet]

                except KeyError:
                    concreteDecoder = None

                if concreteDecoder:
                    state = stDecodeValue

                else:
                    try:
                        concreteDecoder = tagMap[tagSet[:1]]

                    except KeyError:
                        concreteDecoder = None

                    if concreteDecoder:
                        state = stDecodeValue
                    else:
                        state = stTryAsExplicitTag

                if LOG:
                    LOG('codec %s chosen by a built-in type, decoding %s' % (concreteDecoder and concreteDecoder.__class__.__name__ or "<none>", state is stDecodeValue and 'value' or 'as explicit tag'))
                    debug.scope.push(concreteDecoder is None and '?' or concreteDecoder.protoComponent.__class__.__name__)

            if state is stGetValueDecoderByAsn1Spec:

                if asn1Spec.__class__ is tagmap.TagMap:
                    try:
                        chosenSpec = asn1Spec[tagSet]

                    except KeyError:
                        chosenSpec = None

                    if LOG:
                        LOG('candidate ASN.1 spec is a map of:')

                        for firstOctet, v in asn1Spec.presentTypes.items():
                            LOG('  %s -> %s' % (firstOctet, v.__class__.__name__))

                        if asn1Spec.skipTypes:
                            LOG('but neither of: ')
                            for firstOctet, v in asn1Spec.skipTypes.items():
                                LOG('  %s -> %s' % (firstOctet, v.__class__.__name__))
                        LOG('new candidate ASN.1 spec is %s, chosen by %s' % (chosenSpec is None and '<none>' or chosenSpec.prettyPrintType(), tagSet))

                elif tagSet == asn1Spec.tagSet or tagSet in asn1Spec.tagMap:
                    chosenSpec = asn1Spec
                    if LOG:
                        LOG('candidate ASN.1 spec is %s' % asn1Spec.__class__.__name__)

                else:
                    chosenSpec = None

                if chosenSpec is not None:
                    try:
                        # ambiguous type or just faster codec lookup
                        concreteDecoder = typeMap[chosenSpec.typeId]

                        if LOG:
                            LOG('value decoder chosen for an ambiguous type by type ID %s' % (chosenSpec.typeId,))

                    except KeyError:
                        # use base type for codec lookup to recover untagged types
                        baseTagSet = tag.TagSet(chosenSpec.tagSet.baseTag,  chosenSpec.tagSet.baseTag)
                        try:
                            # base type or tagged subtype
                            concreteDecoder = tagMap[baseTagSet]

                            if LOG:
                                LOG('value decoder chosen by base %s' % (baseTagSet,))

                        except KeyError:
                            concreteDecoder = None

                    if concreteDecoder:
                        asn1Spec = chosenSpec
                        state = stDecodeValue

                    else:
                        state = stTryAsExplicitTag

                else:
                    concreteDecoder = None
                    state = stTryAsExplicitTag

                if LOG:
                    LOG('codec %s chosen by ASN.1 spec, decoding %s' % (state is stDecodeValue and concreteDecoder.__class__.__name__ or "<none>", state is stDecodeValue and 'value' or 'as explicit tag'))
                    debug.scope.push(chosenSpec is None and '?' or chosenSpec.__class__.__name__)

            if state is stDecodeValue:
                if not options.get('recursiveFlag', True) and not substrateFun:  # deprecate this
                    substrateFun = lambda a, b, c: (a, b[:c])

                options.update(fullSubstrate=fullSubstrate)

                if length == -1:  # indef length
                    value, substrate = concreteDecoder.indefLenValueDecoder(
                        substrate, asn1Spec,
                        tagSet, length, stGetValueDecoder,
                        self, substrateFun,
                        **options
                    )

                else:
                    value, substrate = concreteDecoder.valueDecoder(
                        substrate, asn1Spec,
                        tagSet, length, stGetValueDecoder,
                        self, substrateFun,
                        **options
                    )

                if LOG:
                    LOG('codec %s yields type %s, value:\n%s\n...remaining substrate is: %s' % (concreteDecoder.__class__.__name__, value.__class__.__name__, isinstance(value, base.Asn1Item) and value.prettyPrint() or value, substrate and debug.hexdump(substrate) or '<none>'))

                state = stStop
                break

            if state is stTryAsExplicitTag:
                if (tagSet and
                        tagSet[0].tagFormat == tag.tagFormatConstructed and
                        tagSet[0].tagClass != tag.tagClassUniversal):
                    # Assume explicit tagging
                    concreteDecoder = explicitTagDecoder
                    state = stDecodeValue

                else:
                    concreteDecoder = None
                    state = self.defaultErrorState

                if LOG:
                    LOG('codec %s chosen, decoding %s' % (concreteDecoder and concreteDecoder.__class__.__name__ or "<none>", state is stDecodeValue and 'value' or 'as failure'))

            if state is stDumpRawValue:
                concreteDecoder = self.defaultRawDecoder

                if LOG:
                    LOG('codec %s chosen, decoding value' % concreteDecoder.__class__.__name__)

                state = stDecodeValue

            if state is stErrorCondition:
                raise error.PyAsn1Error(
                    '%s not in asn1Spec: %r' % (tagSet, asn1Spec)
                )

        if LOG:
            debug.scope.pop()
            LOG('decoder left scope %s, call completed' % debug.scope)

        return value, substrate


#: Turns BER octet stream into an ASN.1 object.
#:
#: Takes BER octet-stream and decode it into an ASN.1 object
#: (e.g. :py:class:`~pyasn1.type.base.PyAsn1Item` derivative) which
#: may be a scalar or an arbitrary nested structure.
#:
#: Parameters
#: ----------
#: substrate: :py:class:`bytes` (Python 3) or :py:class:`str` (Python 2)
#:     BER octet-stream
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
#:     A tuple of pyasn1 object recovered from BER substrate (:py:class:`~pyasn1.type.base.PyAsn1Item` derivative)
#:     and the unprocessed trailing portion of the *substrate* (may be empty)
#:
#: Raises
#: ------
#: ~pyasn1.error.PyAsn1Error, ~pyasn1.error.SubstrateUnderrunError
#:     On decoding errors
#:
#: Examples
#: --------
#: Decode BER serialisation without ASN.1 schema
#:
#: .. code-block:: pycon
#:
#:    >>> s, _ = decode(b'0\t\x02\x01\x01\x02\x01\x02\x02\x01\x03')
#:    >>> str(s)
#:    SequenceOf:
#:     1 2 3
#:
#: Decode BER serialisation with ASN.1 schema
#:
#: .. code-block:: pycon
#:
#:    >>> seq = SequenceOf(componentType=Integer())
#:    >>> s, _ = decode(b'0\t\x02\x01\x01\x02\x01\x02\x02\x01\x03', asn1Spec=seq)
#:    >>> str(s)
#:    SequenceOf:
#:     1 2 3
#:
decode = Decoder(tagMap, typeMap)

# XXX
# non-recursive decoding; return position rather than substrate
