# General utilities for MAPI and MAPI objects.
# We used to use these old names from the 'types' module...
TupleType = tuple
ListType = list
IntType = int
import pythoncom
from pywintypes import TimeType

from . import mapi, mapitags

prTable = {}


def GetPropTagName(pt):
    if not prTable:
        for name, value in mapitags.__dict__.items():
            if name[:3] == "PR_":
                # Store both the full ID (including type) and just the ID.
                # This is so PR_FOO_A and PR_FOO_W are still differentiated,
                # but should we get a PT_FOO with PT_ERROR set, we fallback
                # to the ID.

                # String types should have 3 definitions in mapitags.py
                # PR_BODY	= PROP_TAG( PT_TSTRING,	4096)
                # PR_BODY_W	= PROP_TAG( PT_UNICODE, 4096)
                # PR_BODY_A	= PROP_TAG( PT_STRING8, 4096)
                # The following change ensures a lookup using only the the
                # property id returns the conditional default.

                # PT_TSTRING is a conditional assignment for either PT_UNICODE or
                # PT_STRING8 and should not be returned during a lookup.

                if (
                    mapitags.PROP_TYPE(value) == mapitags.PT_UNICODE
                    or mapitags.PROP_TYPE(value) == mapitags.PT_STRING8
                ):
                    if name[-2:] == "_A" or name[-2:] == "_W":
                        prTable[value] = name
                    else:
                        prTable[mapitags.PROP_ID(value)] = name

                else:
                    prTable[value] = name
                    prTable[mapitags.PROP_ID(value)] = name

    try:
        try:
            return prTable[pt]
        except KeyError:
            # Can't find it exactly - see if the raw ID exists.
            return prTable[mapitags.PROP_ID(pt)]
    except KeyError:
        # god-damn bullshit hex() warnings: I don't see a way to get the
        # old behaviour without a warning!!
        ret = hex(int(pt))
        # -0x8000000L -> 0x80000000
        if ret[0] == "-":
            ret = ret[1:]
        if ret[-1] == "L":
            ret = ret[:-1]
        return ret


mapiErrorTable = {}


def GetScodeString(hr):
    if not mapiErrorTable:
        for name, value in mapi.__dict__.items():
            if name[:7] in ["MAPI_E_", "MAPI_W_"]:
                mapiErrorTable[value] = name
    return mapiErrorTable.get(hr, pythoncom.GetScodeString(hr))


ptTable = {}


def GetMapiTypeName(propType, rawType=True):
    """Given a mapi type flag, return a string description of the type"""
    if not ptTable:
        for name, value in mapitags.__dict__.items():
            if name[:3] == "PT_":
                # PT_TSTRING is a conditional assignment
                # for either PT_UNICODE or PT_STRING8 and
                # should not be returned during a lookup.
                if name in ["PT_TSTRING", "PT_MV_TSTRING"]:
                    continue
                ptTable[value] = name

    if rawType:
        propType = propType & ~mapitags.MV_FLAG
    return ptTable.get(propType, str(hex(propType)))


def GetProperties(obj, propList):
    """Given a MAPI object and a list of properties, return a list of property values.

    Allows a single property to be passed, and the result is a single object.

    Each request property can be an integer or a string.  Of a string, it is
    automatically converted to an integer via the GetIdsFromNames function.

    If the property fetch fails, the result is None.
    """
    bRetList = 1
    if type(propList) not in [TupleType, ListType]:
        bRetList = 0
        propList = (propList,)
    realPropList = []
    rc = []
    for prop in propList:
        if type(prop) != IntType:  # Integer
            props = ((mapi.PS_PUBLIC_STRINGS, prop),)
            propIds = obj.GetIDsFromNames(props, 0)
            prop = mapitags.PROP_TAG(
                mapitags.PT_UNSPECIFIED, mapitags.PROP_ID(propIds[0])
            )
        realPropList.append(prop)

    hr, data = obj.GetProps(realPropList, 0)
    if hr != 0:
        data = None
        return None
    if bRetList:
        return [v[1] for v in data]
    else:
        return data[0][1]


def GetAllProperties(obj, make_tag_names=True):
    tags = obj.GetPropList(0)
    hr, data = obj.GetProps(tags)
    ret = []
    for tag, val in data:
        if make_tag_names:
            hr, tags, array = obj.GetNamesFromIDs((tag,))
            if type(array[0][1]) == type(""):
                name = array[0][1]
            else:
                name = GetPropTagName(tag)
        else:
            name = tag
        ret.append((name, val))
    return ret


_MapiTypeMap = {
    type(0.0): mapitags.PT_DOUBLE,
    type(0): mapitags.PT_I4,
    type("".encode("ascii")): mapitags.PT_STRING8,  # bytes
    type(""): mapitags.PT_UNICODE,  # str
    type(None): mapitags.PT_UNSPECIFIED,
    # In Python 2.2.2, bool isn't a distinct type (type(1==1) is type(0)).
    # (markh thinks the above is trying to say that in 2020, we probably *do*
    # want bool in this map? :)
}


def SetPropertyValue(obj, prop, val):
    if type(prop) != IntType:
        props = ((mapi.PS_PUBLIC_STRINGS, prop),)
        propIds = obj.GetIDsFromNames(props, mapi.MAPI_CREATE)
        if val == (1 == 1) or val == (1 == 0):
            type_tag = mapitags.PT_BOOLEAN
        else:
            type_tag = _MapiTypeMap.get(type(val))
            if type_tag is None:
                raise ValueError(
                    "Don't know what to do with '%r' ('%s')" % (val, type(val))
                )
        prop = mapitags.PROP_TAG(type_tag, mapitags.PROP_ID(propIds[0]))
    if val is None:
        # Delete the property
        obj.DeleteProps((prop,))
    else:
        obj.SetProps(((prop, val),))


def SetProperties(msg, propDict):
    """Given a Python dictionary, set the objects properties.

    If the dictionary key is a string, then a property ID is queried
    otherwise the ID is assumed native.

    Coded for maximum efficiency wrt server calls - ie, maximum of
    2 calls made to the object, regardless of the dictionary contents
    (only 1 if dictionary full of int keys)
    """

    newProps = []
    # First pass over the properties we should get IDs for.
    for key, val in propDict.items():
        if type(key) == str:
            newProps.append((mapi.PS_PUBLIC_STRINGS, key))
    # Query for the new IDs
    if newProps:
        newIds = msg.GetIDsFromNames(newProps, mapi.MAPI_CREATE)
    newIdNo = 0
    newProps = []
    for key, val in propDict.items():
        if type(key) == str:
            type_val = type(val)
            if type_val == str:
                tagType = mapitags.PT_UNICODE
            elif type_val == IntType:
                tagType = mapitags.PT_I4
            elif type_val == TimeType:
                tagType = mapitags.PT_SYSTIME
            else:
                raise ValueError(
                    "The type of object %s(%s) can not be written"
                    % (repr(val), type_val)
                )
            key = mapitags.PROP_TAG(tagType, mapitags.PROP_ID(newIds[newIdNo]))
            newIdNo = newIdNo + 1
        newProps.append((key, val))
    msg.SetProps(newProps)
