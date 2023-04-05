#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
import sys

from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap

__all__ = ['NamedType', 'OptionalNamedType', 'DefaultedNamedType',
           'NamedTypes']

try:
    any

except NameError:
    any = lambda x: bool(filter(bool, x))


class NamedType(object):
    """Create named field object for a constructed ASN.1 type.

    The |NamedType| object represents a single name and ASN.1 type of a constructed ASN.1 type.

    |NamedType| objects are immutable and duck-type Python :class:`tuple` objects
    holding *name* and *asn1Object* components.

    Parameters
    ----------
    name: :py:class:`str`
        Field name

    asn1Object:
        ASN.1 type object
    """
    isOptional = False
    isDefaulted = False

    def __init__(self, name, asn1Object, openType=None):
        self.__name = name
        self.__type = asn1Object
        self.__nameAndType = name, asn1Object
        self.__openType = openType

    def __repr__(self):
        representation = '%s=%r' % (self.name, self.asn1Object)

        if self.openType:
            representation += ', open type %r' % self.openType

        return '<%s object, type %s>' % (
            self.__class__.__name__, representation)

    def __eq__(self, other):
        return self.__nameAndType == other

    def __ne__(self, other):
        return self.__nameAndType != other

    def __lt__(self, other):
        return self.__nameAndType < other

    def __le__(self, other):
        return self.__nameAndType <= other

    def __gt__(self, other):
        return self.__nameAndType > other

    def __ge__(self, other):
        return self.__nameAndType >= other

    def __hash__(self):
        return hash(self.__nameAndType)

    def __getitem__(self, idx):
        return self.__nameAndType[idx]

    def __iter__(self):
        return iter(self.__nameAndType)

    @property
    def name(self):
        return self.__name

    @property
    def asn1Object(self):
        return self.__type

    @property
    def openType(self):
        return self.__openType

    # Backward compatibility

    def getName(self):
        return self.name

    def getType(self):
        return self.asn1Object


class OptionalNamedType(NamedType):
    __doc__ = NamedType.__doc__

    isOptional = True


class DefaultedNamedType(NamedType):
    __doc__ = NamedType.__doc__

    isDefaulted = True


class NamedTypes(object):
    """Create a collection of named fields for a constructed ASN.1 type.

    The NamedTypes object represents a collection of named fields of a constructed ASN.1 type.

    *NamedTypes* objects are immutable and duck-type Python :class:`dict` objects
    holding *name* as keys and ASN.1 type object as values.

    Parameters
    ----------
    *namedTypes: :class:`~pyasn1.type.namedtype.NamedType`

    Examples
    --------

    .. code-block:: python

        class Description(Sequence):
            '''
            ASN.1 specification:

            Description ::= SEQUENCE {
                surname    IA5String,
                first-name IA5String OPTIONAL,
                age        INTEGER DEFAULT 40
            }
            '''
            componentType = NamedTypes(
                NamedType('surname', IA5String()),
                OptionalNamedType('first-name', IA5String()),
                DefaultedNamedType('age', Integer(40))
            )

        descr = Description()
        descr['surname'] = 'Smith'
        descr['first-name'] = 'John'
    """
    def __init__(self, *namedTypes, **kwargs):
        self.__namedTypes = namedTypes
        self.__namedTypesLen = len(self.__namedTypes)
        self.__minTagSet = self.__computeMinTagSet()
        self.__nameToPosMap = self.__computeNameToPosMap()
        self.__tagToPosMap = self.__computeTagToPosMap()
        self.__ambiguousTypes = 'terminal' not in kwargs and self.__computeAmbiguousTypes() or {}
        self.__uniqueTagMap = self.__computeTagMaps(unique=True)
        self.__nonUniqueTagMap = self.__computeTagMaps(unique=False)
        self.__hasOptionalOrDefault = any([True for namedType in self.__namedTypes
                                           if namedType.isDefaulted or namedType.isOptional])
        self.__hasOpenTypes = any([True for namedType in self.__namedTypes
                                   if namedType.openType])

        self.__requiredComponents = frozenset(
                [idx for idx, nt in enumerate(self.__namedTypes) if not nt.isOptional and not nt.isDefaulted]
            )
        self.__keys = frozenset([namedType.name for namedType in self.__namedTypes])
        self.__values = tuple([namedType.asn1Object for namedType in self.__namedTypes])
        self.__items = tuple([(namedType.name, namedType.asn1Object) for namedType in self.__namedTypes])

    def __repr__(self):
        representation = ', '.join(['%r' % x for x in self.__namedTypes])
        return '<%s object, types %s>' % (
            self.__class__.__name__, representation)

    def __eq__(self, other):
        return self.__namedTypes == other

    def __ne__(self, other):
        return self.__namedTypes != other

    def __lt__(self, other):
        return self.__namedTypes < other

    def __le__(self, other):
        return self.__namedTypes <= other

    def __gt__(self, other):
        return self.__namedTypes > other

    def __ge__(self, other):
        return self.__namedTypes >= other

    def __hash__(self):
        return hash(self.__namedTypes)

    def __getitem__(self, idx):
        try:
            return self.__namedTypes[idx]

        except TypeError:
            return self.__namedTypes[self.__nameToPosMap[idx]]

    def __contains__(self, key):
        return key in self.__nameToPosMap

    def __iter__(self):
        return (x[0] for x in self.__namedTypes)

    if sys.version_info[0] <= 2:
        def __nonzero__(self):
            return self.__namedTypesLen > 0
    else:
        def __bool__(self):
            return self.__namedTypesLen > 0

    def __len__(self):
        return self.__namedTypesLen

    # Python dict protocol

    def values(self):
        return self.__values

    def keys(self):
        return self.__keys

    def items(self):
        return self.__items

    def clone(self):
        return self.__class__(*self.__namedTypes)

    class PostponedError(object):
        def __init__(self, errorMsg):
            self.__errorMsg = errorMsg

        def __getitem__(self, item):
            raise  error.PyAsn1Error(self.__errorMsg)

    def __computeTagToPosMap(self):
        tagToPosMap = {}
        for idx, namedType in enumerate(self.__namedTypes):
            tagMap = namedType.asn1Object.tagMap
            if isinstance(tagMap, NamedTypes.PostponedError):
                return tagMap
            if not tagMap:
                continue
            for _tagSet in tagMap.presentTypes:
                if _tagSet in tagToPosMap:
                    return NamedTypes.PostponedError('Duplicate component tag %s at %s' % (_tagSet, namedType))
                tagToPosMap[_tagSet] = idx

        return tagToPosMap

    def __computeNameToPosMap(self):
        nameToPosMap = {}
        for idx, namedType in enumerate(self.__namedTypes):
            if namedType.name in nameToPosMap:
                return NamedTypes.PostponedError('Duplicate component name %s at %s' % (namedType.name, namedType))
            nameToPosMap[namedType.name] = idx

        return nameToPosMap

    def __computeAmbiguousTypes(self):
        ambiguousTypes = {}
        partialAmbiguousTypes = ()
        for idx, namedType in reversed(tuple(enumerate(self.__namedTypes))):
            if namedType.isOptional or namedType.isDefaulted:
                partialAmbiguousTypes = (namedType,) + partialAmbiguousTypes
            else:
                partialAmbiguousTypes = (namedType,)
            if len(partialAmbiguousTypes) == len(self.__namedTypes):
                ambiguousTypes[idx] = self
            else:
                ambiguousTypes[idx] = NamedTypes(*partialAmbiguousTypes, **dict(terminal=True))
        return ambiguousTypes

    def getTypeByPosition(self, idx):
        """Return ASN.1 type object by its position in fields set.

        Parameters
        ----------
        idx: :py:class:`int`
            Field index

        Returns
        -------
        :
            ASN.1 type

        Raises
        ------
        ~pyasn1.error.PyAsn1Error
            If given position is out of fields range
        """
        try:
            return self.__namedTypes[idx].asn1Object

        except IndexError:
            raise error.PyAsn1Error('Type position out of range')

    def getPositionByType(self, tagSet):
        """Return field position by its ASN.1 type.

        Parameters
        ----------
        tagSet: :class:`~pysnmp.type.tag.TagSet`
            ASN.1 tag set distinguishing one ASN.1 type from others.

        Returns
        -------
        : :py:class:`int`
            ASN.1 type position in fields set

        Raises
        ------
        ~pyasn1.error.PyAsn1Error
            If *tagSet* is not present or ASN.1 types are not unique within callee *NamedTypes*
        """
        try:
            return self.__tagToPosMap[tagSet]

        except KeyError:
            raise error.PyAsn1Error('Type %s not found' % (tagSet,))

    def getNameByPosition(self, idx):
        """Return field name by its position in fields set.

        Parameters
        ----------
        idx: :py:class:`idx`
            Field index

        Returns
        -------
        : :py:class:`str`
            Field name

        Raises
        ------
        ~pyasn1.error.PyAsn1Error
            If given field name is not present in callee *NamedTypes*
        """
        try:
            return self.__namedTypes[idx].name

        except IndexError:
            raise error.PyAsn1Error('Type position out of range')

    def getPositionByName(self, name):
        """Return field position by filed name.

        Parameters
        ----------
        name: :py:class:`str`
            Field name

        Returns
        -------
        : :py:class:`int`
            Field position in fields set

        Raises
        ------
        ~pyasn1.error.PyAsn1Error
            If *name* is not present or not unique within callee *NamedTypes*
        """
        try:
            return self.__nameToPosMap[name]

        except KeyError:
            raise error.PyAsn1Error('Name %s not found' % (name,))

    def getTagMapNearPosition(self, idx):
        """Return ASN.1 types that are allowed at or past given field position.

        Some ASN.1 serialisation allow for skipping optional and defaulted fields.
        Some constructed ASN.1 types allow reordering of the fields. When recovering
        such objects it may be important to know which types can possibly be
        present at any given position in the field sets.

        Parameters
        ----------
        idx: :py:class:`int`
            Field index

        Returns
        -------
        : :class:`~pyasn1.type.tagmap.TagMap`
            Map if ASN.1 types allowed at given field position

        Raises
        ------
        ~pyasn1.error.PyAsn1Error
            If given position is out of fields range
        """
        try:
            return self.__ambiguousTypes[idx].tagMap

        except KeyError:
            raise error.PyAsn1Error('Type position out of range')

    def getPositionNearType(self, tagSet, idx):
        """Return the closest field position where given ASN.1 type is allowed.

        Some ASN.1 serialisation allow for skipping optional and defaulted fields.
        Some constructed ASN.1 types allow reordering of the fields. When recovering
        such objects it may be important to know at which field position, in field set,
        given *tagSet* is allowed at or past *idx* position.

        Parameters
        ----------
        tagSet: :class:`~pyasn1.type.tag.TagSet`
           ASN.1 type which field position to look up

        idx: :py:class:`int`
            Field position at or past which to perform ASN.1 type look up

        Returns
        -------
        : :py:class:`int`
            Field position in fields set

        Raises
        ------
        ~pyasn1.error.PyAsn1Error
            If *tagSet* is not present or not unique within callee *NamedTypes*
            or *idx* is out of fields range
        """
        try:
            return idx + self.__ambiguousTypes[idx].getPositionByType(tagSet)

        except KeyError:
            raise error.PyAsn1Error('Type position out of range')

    def __computeMinTagSet(self):
        minTagSet = None
        for namedType in self.__namedTypes:
            asn1Object = namedType.asn1Object

            try:
                tagSet = asn1Object.minTagSet

            except AttributeError:
                tagSet = asn1Object.tagSet

            if minTagSet is None or tagSet < minTagSet:
                minTagSet = tagSet

        return minTagSet or tag.TagSet()

    @property
    def minTagSet(self):
        """Return the minimal TagSet among ASN.1 type in callee *NamedTypes*.

        Some ASN.1 types/serialisation protocols require ASN.1 types to be
        arranged based on their numerical tag value. The *minTagSet* property
        returns that.

        Returns
        -------
        : :class:`~pyasn1.type.tagset.TagSet`
            Minimal TagSet among ASN.1 types in callee *NamedTypes*
        """
        return self.__minTagSet

    def __computeTagMaps(self, unique):
        presentTypes = {}
        skipTypes = {}
        defaultType = None
        for namedType in self.__namedTypes:
            tagMap = namedType.asn1Object.tagMap
            if isinstance(tagMap, NamedTypes.PostponedError):
                return tagMap
            for tagSet in tagMap:
                if unique and tagSet in presentTypes:
                    return NamedTypes.PostponedError('Non-unique tagSet %s of %s at %s' % (tagSet, namedType, self))
                presentTypes[tagSet] = namedType.asn1Object
            skipTypes.update(tagMap.skipTypes)

            if defaultType is None:
                defaultType = tagMap.defaultType
            elif tagMap.defaultType is not None:
                return NamedTypes.PostponedError('Duplicate default ASN.1 type at %s' % (self,))

        return tagmap.TagMap(presentTypes, skipTypes, defaultType)

    @property
    def tagMap(self):
        """Return a *TagMap* object from tags and types recursively.

        Return a :class:`~pyasn1.type.tagmap.TagMap` object by
        combining tags from *TagMap* objects of children types and
        associating them with their immediate child type.

        Example
        -------
        .. code-block:: python

           OuterType ::= CHOICE {
               innerType INTEGER
           }

        Calling *.tagMap* on *OuterType* will yield a map like this:

        .. code-block:: python

           Integer.tagSet -> Choice
        """
        return self.__nonUniqueTagMap

    @property
    def tagMapUnique(self):
        """Return a *TagMap* object from unique tags and types recursively.

        Return a :class:`~pyasn1.type.tagmap.TagMap` object by
        combining tags from *TagMap* objects of children types and
        associating them with their immediate child type.

        Example
        -------
        .. code-block:: python

           OuterType ::= CHOICE {
               innerType INTEGER
           }

        Calling *.tagMapUnique* on *OuterType* will yield a map like this:

        .. code-block:: python

           Integer.tagSet -> Choice

        Note
        ----

        Duplicate *TagSet* objects found in the tree of children
        types would cause error.
        """
        return self.__uniqueTagMap

    @property
    def hasOptionalOrDefault(self):
        return self.__hasOptionalOrDefault

    @property
    def hasOpenTypes(self):
        return self.__hasOpenTypes

    @property
    def namedTypes(self):
        return tuple(self.__namedTypes)

    @property
    def requiredComponents(self):
        return self.__requiredComponents
