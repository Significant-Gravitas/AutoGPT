#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from pyasn1 import error

__all__ = ['tagClassUniversal', 'tagClassApplication', 'tagClassContext',
           'tagClassPrivate', 'tagFormatSimple', 'tagFormatConstructed',
           'tagCategoryImplicit', 'tagCategoryExplicit',
           'tagCategoryUntagged', 'Tag', 'TagSet']

#: Identifier for ASN.1 class UNIVERSAL
tagClassUniversal = 0x00

#: Identifier for ASN.1 class APPLICATION
tagClassApplication = 0x40

#: Identifier for ASN.1 class context-specific
tagClassContext = 0x80

#: Identifier for ASN.1 class private
tagClassPrivate = 0xC0

#: Identifier for "simple" ASN.1 structure (e.g. scalar)
tagFormatSimple = 0x00

#: Identifier for "constructed" ASN.1 structure (e.g. may have inner components)
tagFormatConstructed = 0x20

tagCategoryImplicit = 0x01
tagCategoryExplicit = 0x02
tagCategoryUntagged = 0x04


class Tag(object):
    """Create ASN.1 tag

    Represents ASN.1 tag that can be attached to a ASN.1 type to make
    types distinguishable from each other.

    *Tag* objects are immutable and duck-type Python :class:`tuple` objects
    holding three integer components of a tag.

    Parameters
    ----------
    tagClass: :py:class:`int`
        Tag *class* value

    tagFormat: :py:class:`int`
        Tag *format* value

    tagId: :py:class:`int`
        Tag ID value
    """
    def __init__(self, tagClass, tagFormat, tagId):
        if tagId < 0:
            raise error.PyAsn1Error('Negative tag ID (%s) not allowed' % tagId)
        self.__tagClass = tagClass
        self.__tagFormat = tagFormat
        self.__tagId = tagId
        self.__tagClassId = tagClass, tagId
        self.__hash = hash(self.__tagClassId)

    def __repr__(self):
        representation = '[%s:%s:%s]' % (
            self.__tagClass, self.__tagFormat, self.__tagId)
        return '<%s object, tag %s>' % (
            self.__class__.__name__, representation)

    def __eq__(self, other):
        return self.__tagClassId == other

    def __ne__(self, other):
        return self.__tagClassId != other

    def __lt__(self, other):
        return self.__tagClassId < other

    def __le__(self, other):
        return self.__tagClassId <= other

    def __gt__(self, other):
        return self.__tagClassId > other

    def __ge__(self, other):
        return self.__tagClassId >= other

    def __hash__(self):
        return self.__hash

    def __getitem__(self, idx):
        if idx == 0:
            return self.__tagClass
        elif idx == 1:
            return self.__tagFormat
        elif idx == 2:
            return self.__tagId
        else:
            raise IndexError()

    def __iter__(self):
        yield self.__tagClass
        yield self.__tagFormat
        yield self.__tagId

    def __and__(self, otherTag):
        return self.__class__(self.__tagClass & otherTag.tagClass,
                              self.__tagFormat & otherTag.tagFormat,
                              self.__tagId & otherTag.tagId)

    def __or__(self, otherTag):
        return self.__class__(self.__tagClass | otherTag.tagClass,
                              self.__tagFormat | otherTag.tagFormat,
                              self.__tagId | otherTag.tagId)

    @property
    def tagClass(self):
        """ASN.1 tag class

        Returns
        -------
        : :py:class:`int`
            Tag class
        """
        return self.__tagClass

    @property
    def tagFormat(self):
        """ASN.1 tag format

        Returns
        -------
        : :py:class:`int`
            Tag format
        """
        return self.__tagFormat

    @property
    def tagId(self):
        """ASN.1 tag ID

        Returns
        -------
        : :py:class:`int`
            Tag ID
        """
        return self.__tagId


class TagSet(object):
    """Create a collection of ASN.1 tags

    Represents a combination of :class:`~pyasn1.type.tag.Tag` objects
    that can be attached to a ASN.1 type to make types distinguishable
    from each other.

    *TagSet* objects are immutable and duck-type Python :class:`tuple` objects
    holding arbitrary number of :class:`~pyasn1.type.tag.Tag` objects.

    Parameters
    ----------
    baseTag: :class:`~pyasn1.type.tag.Tag`
        Base *Tag* object. This tag survives IMPLICIT tagging.

    *superTags: :class:`~pyasn1.type.tag.Tag`
        Additional *Tag* objects taking part in subtyping.

    Examples
    --------
    .. code-block:: python

        class OrderNumber(NumericString):
            '''
            ASN.1 specification

            Order-number ::=
                [APPLICATION 5] IMPLICIT NumericString
            '''
            tagSet = NumericString.tagSet.tagImplicitly(
                Tag(tagClassApplication, tagFormatSimple, 5)
            )

        orderNumber = OrderNumber('1234')
    """
    def __init__(self, baseTag=(), *superTags):
        self.__baseTag = baseTag
        self.__superTags = superTags
        self.__superTagsClassId = tuple(
            [(superTag.tagClass, superTag.tagId) for superTag in superTags]
        )
        self.__lenOfSuperTags = len(superTags)
        self.__hash = hash(self.__superTagsClassId)

    def __repr__(self):
        representation = '-'.join(['%s:%s:%s' % (x.tagClass, x.tagFormat, x.tagId)
                                   for x in self.__superTags])
        if representation:
            representation = 'tags ' + representation
        else:
            representation = 'untagged'

        return '<%s object, %s>' % (self.__class__.__name__, representation)

    def __add__(self, superTag):
        return self.__class__(self.__baseTag, *self.__superTags + (superTag,))

    def __radd__(self, superTag):
        return self.__class__(self.__baseTag, *(superTag,) + self.__superTags)

    def __getitem__(self, i):
        if i.__class__ is slice:
            return self.__class__(self.__baseTag, *self.__superTags[i])
        else:
            return self.__superTags[i]

    def __eq__(self, other):
        return self.__superTagsClassId == other

    def __ne__(self, other):
        return self.__superTagsClassId != other

    def __lt__(self, other):
        return self.__superTagsClassId < other

    def __le__(self, other):
        return self.__superTagsClassId <= other

    def __gt__(self, other):
        return self.__superTagsClassId > other

    def __ge__(self, other):
        return self.__superTagsClassId >= other

    def __hash__(self):
        return self.__hash

    def __len__(self):
        return self.__lenOfSuperTags

    @property
    def baseTag(self):
        """Return base ASN.1 tag

        Returns
        -------
        : :class:`~pyasn1.type.tag.Tag`
            Base tag of this *TagSet*
        """
        return self.__baseTag

    @property
    def superTags(self):
        """Return ASN.1 tags

        Returns
        -------
        : :py:class:`tuple`
            Tuple of :class:`~pyasn1.type.tag.Tag` objects that this *TagSet* contains
        """
        return self.__superTags

    def tagExplicitly(self, superTag):
        """Return explicitly tagged *TagSet*

        Create a new *TagSet* representing callee *TagSet* explicitly tagged
        with passed tag(s). With explicit tagging mode, new tags are appended
        to existing tag(s).

        Parameters
        ----------
        superTag: :class:`~pyasn1.type.tag.Tag`
            *Tag* object to tag this *TagSet*

        Returns
        -------
        : :class:`~pyasn1.type.tag.TagSet`
            New *TagSet* object
        """
        if superTag.tagClass == tagClassUniversal:
            raise error.PyAsn1Error("Can't tag with UNIVERSAL class tag")
        if superTag.tagFormat != tagFormatConstructed:
            superTag = Tag(superTag.tagClass, tagFormatConstructed, superTag.tagId)
        return self + superTag

    def tagImplicitly(self, superTag):
        """Return implicitly tagged *TagSet*

        Create a new *TagSet* representing callee *TagSet* implicitly tagged
        with passed tag(s). With implicit tagging mode, new tag(s) replace the
        last existing tag.

        Parameters
        ----------
        superTag: :class:`~pyasn1.type.tag.Tag`
            *Tag* object to tag this *TagSet*

        Returns
        -------
        : :class:`~pyasn1.type.tag.TagSet`
            New *TagSet* object
        """
        if self.__superTags:
            superTag = Tag(superTag.tagClass, self.__superTags[-1].tagFormat, superTag.tagId)
        return self[:-1] + superTag

    def isSuperTagSetOf(self, tagSet):
        """Test type relationship against given *TagSet*

        The callee is considered to be a supertype of given *TagSet*
        tag-wise if all tags in *TagSet* are present in the callee and
        they are in the same order.

        Parameters
        ----------
        tagSet: :class:`~pyasn1.type.tag.TagSet`
            *TagSet* object to evaluate against the callee

        Returns
        -------
        : :py:class:`bool`
            :obj:`True` if callee is a supertype of *tagSet*
        """
        if len(tagSet) < self.__lenOfSuperTags:
            return False
        return self.__superTags == tagSet[:self.__lenOfSuperTags]

    # Backward compatibility

    def getBaseTag(self):
        return self.__baseTag

def initTagSet(tag):
    return TagSet(tag, tag)
