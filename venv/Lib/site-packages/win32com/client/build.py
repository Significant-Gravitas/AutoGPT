"""Contains knowledge to build a COM object definition.

This module is used by both the @dynamic@ and @makepy@ modules to build
all knowledge of a COM object.

This module contains classes which contain the actual knowledge of the object.
This include parameter and return type information, the COM dispid and CLSID, etc.

Other modules may use this information to generate .py files, use the information
dynamically, or possibly even generate .html documentation for objects.
"""

#
# NOTES: DispatchItem and MapEntry used by dynamic.py.
#        the rest is used by makepy.py
#
#        OleItem, DispatchItem, MapEntry, BuildCallList() is used by makepy

import datetime
import string
import sys
from keyword import iskeyword

import pythoncom
import winerror
from pywintypes import TimeType


# It isn't really clear what the quoting rules are in a C/IDL string and
# literals like a quote char and backslashes makes life a little painful to
# always render the string perfectly - so just punt and fall-back to a repr()
def _makeDocString(s):
    if sys.version_info < (3,):
        s = s.encode("mbcs")
    return repr(s)


error = "PythonCOM.Client.Build error"


class NotSupportedException(Exception):
    pass  # Raised when we cant support a param type.


DropIndirection = "DropIndirection"

NoTranslateTypes = [
    pythoncom.VT_BOOL,
    pythoncom.VT_CLSID,
    pythoncom.VT_CY,
    pythoncom.VT_DATE,
    pythoncom.VT_DECIMAL,
    pythoncom.VT_EMPTY,
    pythoncom.VT_ERROR,
    pythoncom.VT_FILETIME,
    pythoncom.VT_HRESULT,
    pythoncom.VT_I1,
    pythoncom.VT_I2,
    pythoncom.VT_I4,
    pythoncom.VT_I8,
    pythoncom.VT_INT,
    pythoncom.VT_NULL,
    pythoncom.VT_R4,
    pythoncom.VT_R8,
    pythoncom.VT_NULL,
    pythoncom.VT_STREAM,
    pythoncom.VT_UI1,
    pythoncom.VT_UI2,
    pythoncom.VT_UI4,
    pythoncom.VT_UI8,
    pythoncom.VT_UINT,
    pythoncom.VT_VOID,
]

NoTranslateMap = {}
for v in NoTranslateTypes:
    NoTranslateMap[v] = None


class MapEntry:
    "Simple holder for named attibutes - items in a map."

    def __init__(
        self,
        desc_or_id,
        names=None,
        doc=None,
        resultCLSID=pythoncom.IID_NULL,
        resultDoc=None,
        hidden=0,
    ):
        if type(desc_or_id) == type(0):
            self.dispid = desc_or_id
            self.desc = None
        else:
            self.dispid = desc_or_id[0]
            self.desc = desc_or_id

        self.names = names
        self.doc = doc
        self.resultCLSID = resultCLSID
        self.resultDocumentation = resultDoc
        self.wasProperty = (
            0  # Have I been transformed into a function so I can pass args?
        )
        self.hidden = hidden

    def __repr__(self):
        return (
            "MapEntry(dispid={s.dispid}, desc={s.desc}, names={s.names}, doc={s.doc!r}, "
            "resultCLSID={s.resultCLSID}, resultDocumentation={s.resultDocumentation}, "
            "wasProperty={s.wasProperty}, hidden={s.hidden}"
        ).format(s=self)

    def GetResultCLSID(self):
        rc = self.resultCLSID
        if rc == pythoncom.IID_NULL:
            return None
        return rc

    # Return a string, suitable for output - either "'{...}'" or "None"
    def GetResultCLSIDStr(self):
        rc = self.GetResultCLSID()
        if rc is None:
            return "None"
        return repr(
            str(rc)
        )  # Convert the IID object to a string, then to a string in a string.

    def GetResultName(self):
        if self.resultDocumentation is None:
            return None
        return self.resultDocumentation[0]


class OleItem:
    typename = "OleItem"

    def __init__(self, doc=None):
        self.doc = doc
        if self.doc:
            self.python_name = MakePublicAttributeName(self.doc[0])
        else:
            self.python_name = None
        self.bWritten = 0
        self.bIsDispatch = 0
        self.bIsSink = 0
        self.clsid = None
        self.co_class = None


class DispatchItem(OleItem):
    typename = "DispatchItem"

    def __init__(self, typeinfo=None, attr=None, doc=None, bForUser=1):
        OleItem.__init__(self, doc)
        self.propMap = {}
        self.propMapGet = {}
        self.propMapPut = {}
        self.mapFuncs = {}
        self.defaultDispatchName = None
        self.hidden = 0

        if typeinfo:
            self.Build(typeinfo, attr, bForUser)

    def _propMapPutCheck_(self, key, item):
        ins, outs, opts = self.CountInOutOptArgs(item.desc[2])
        if ins > 1:  # if a Put property takes more than 1 arg:
            if opts + 1 == ins or ins == item.desc[6] + 1:
                newKey = "Set" + key
                deleteExisting = 0  # This one is still OK
            else:
                deleteExisting = 1  # No good to us
                if key in self.mapFuncs or key in self.propMapGet:
                    newKey = "Set" + key
                else:
                    newKey = key
            item.wasProperty = 1
            self.mapFuncs[newKey] = item
            if deleteExisting:
                del self.propMapPut[key]

    def _propMapGetCheck_(self, key, item):
        ins, outs, opts = self.CountInOutOptArgs(item.desc[2])
        if ins > 0:  # if a Get property takes _any_ in args:
            if item.desc[6] == ins or ins == opts:
                newKey = "Get" + key
                deleteExisting = 0  # This one is still OK
            else:
                deleteExisting = 1  # No good to us
                if key in self.mapFuncs:
                    newKey = "Get" + key
                else:
                    newKey = key
            item.wasProperty = 1
            self.mapFuncs[newKey] = item
            if deleteExisting:
                del self.propMapGet[key]

    def _AddFunc_(self, typeinfo, fdesc, bForUser):
        assert fdesc.desckind == pythoncom.DESCKIND_FUNCDESC
        id = fdesc.memid
        funcflags = fdesc.wFuncFlags
        try:
            names = typeinfo.GetNames(id)
            name = names[0]
        except pythoncom.ole_error:
            name = ""
            names = None

        doc = None
        try:
            if bForUser:
                doc = typeinfo.GetDocumentation(id)
        except pythoncom.ole_error:
            pass

        if id == 0 and name:
            self.defaultDispatchName = name

        invkind = fdesc.invkind

        # We need to translate any Alias', Enums, structs etc in result and args
        typerepr, flag, defval = fdesc.rettype
        # 		sys.stderr.write("%s result - %s -> " % (name, typerepr))
        typerepr, resultCLSID, resultDoc = _ResolveType(typerepr, typeinfo)
        # 		sys.stderr.write("%s\n" % (typerepr,))
        fdesc.rettype = typerepr, flag, defval, resultCLSID
        # Translate any Alias or Enums in argument list.
        argList = []
        for argDesc in fdesc.args:
            typerepr, flag, defval = argDesc
            # 			sys.stderr.write("%s arg - %s -> " % (name, typerepr))
            arg_type, arg_clsid, arg_doc = _ResolveType(typerepr, typeinfo)
            argDesc = arg_type, flag, defval, arg_clsid
            # 			sys.stderr.write("%s\n" % (argDesc[0],))
            argList.append(argDesc)
        fdesc.args = tuple(argList)

        hidden = (funcflags & pythoncom.FUNCFLAG_FHIDDEN) != 0
        if invkind == pythoncom.INVOKE_PROPERTYGET:
            map = self.propMapGet
        # This is not the best solution, but I dont think there is
        # one without specific "set" syntax.
        # If there is a single PUT or PUTREF, it will function as a property.
        # If there are both, then the PUT remains a property, and the PUTREF
        # gets transformed into a function.
        # (in vb, PUT=="obj=other_obj", PUTREF="set obj=other_obj
        elif invkind in (pythoncom.INVOKE_PROPERTYPUT, pythoncom.INVOKE_PROPERTYPUTREF):
            # Special case
            existing = self.propMapPut.get(name, None)
            if existing is not None:
                if existing.desc[4] == pythoncom.INVOKE_PROPERTYPUT:  # Keep this one
                    map = self.mapFuncs
                    name = "Set" + name
                else:  # Existing becomes a func.
                    existing.wasProperty = 1
                    self.mapFuncs["Set" + name] = existing
                    map = self.propMapPut  # existing gets overwritten below.
            else:
                map = self.propMapPut  # first time weve seen it.

        elif invkind == pythoncom.INVOKE_FUNC:
            map = self.mapFuncs
        else:
            map = None
        if not map is None:
            # 				if map.has_key(name):
            # 					sys.stderr.write("Warning - overwriting existing method/attribute %s\n" % name)
            map[name] = MapEntry(fdesc, names, doc, resultCLSID, resultDoc, hidden)
            # any methods that can't be reached via DISPATCH we return None
            # for, so dynamic dispatch doesnt see it.
            if fdesc.funckind != pythoncom.FUNC_DISPATCH:
                return None
            return (name, map)
        return None

    def _AddVar_(self, typeinfo, vardesc, bForUser):
        ### need pythoncom.VARFLAG_FRESTRICTED ...
        ### then check it
        assert vardesc.desckind == pythoncom.DESCKIND_VARDESC

        if vardesc.varkind == pythoncom.VAR_DISPATCH:
            id = vardesc.memid
            names = typeinfo.GetNames(id)
            # Translate any Alias or Enums in result.
            typerepr, flags, defval = vardesc.elemdescVar
            typerepr, resultCLSID, resultDoc = _ResolveType(typerepr, typeinfo)
            vardesc.elemdescVar = typerepr, flags, defval
            doc = None
            try:
                if bForUser:
                    doc = typeinfo.GetDocumentation(id)
            except pythoncom.ole_error:
                pass

            # handle the enumerator specially
            map = self.propMap
            # Check if the element is hidden.
            hidden = (vardesc.wVarFlags & 0x40) != 0  # VARFLAG_FHIDDEN
            map[names[0]] = MapEntry(
                vardesc, names, doc, resultCLSID, resultDoc, hidden
            )
            return (names[0], map)
        else:
            return None

    def Build(self, typeinfo, attr, bForUser=1):
        self.clsid = attr[0]
        self.bIsDispatch = (attr.wTypeFlags & pythoncom.TYPEFLAG_FDISPATCHABLE) != 0
        if typeinfo is None:
            return
        # Loop over all methods
        for j in range(attr[6]):
            fdesc = typeinfo.GetFuncDesc(j)
            self._AddFunc_(typeinfo, fdesc, bForUser)

        # Loop over all variables (ie, properties)
        for j in range(attr[7]):
            fdesc = typeinfo.GetVarDesc(j)
            self._AddVar_(typeinfo, fdesc, bForUser)

        # Now post-process the maps.  For any "Get" or "Set" properties
        # that have arguments, we must turn them into methods.  If a method
        # of the same name already exists, change the name.
        for key, item in list(self.propMapGet.items()):
            self._propMapGetCheck_(key, item)

        for key, item in list(self.propMapPut.items()):
            self._propMapPutCheck_(key, item)

    def CountInOutOptArgs(self, argTuple):
        "Return tuple counting in/outs/OPTS.  Sum of result may not be len(argTuple), as some args may be in/out."
        ins = out = opts = 0
        for argCheck in argTuple:
            inOut = argCheck[1]
            if inOut == 0:
                ins = ins + 1
                out = out + 1
            else:
                if inOut & pythoncom.PARAMFLAG_FIN:
                    ins = ins + 1
                if inOut & pythoncom.PARAMFLAG_FOPT:
                    opts = opts + 1
                if inOut & pythoncom.PARAMFLAG_FOUT:
                    out = out + 1
        return ins, out, opts

    def MakeFuncMethod(self, entry, name, bMakeClass=1):
        # If we have a type description, and not varargs...
        if entry.desc is not None and (len(entry.desc) < 6 or entry.desc[6] != -1):
            return self.MakeDispatchFuncMethod(entry, name, bMakeClass)
        else:
            return self.MakeVarArgsFuncMethod(entry, name, bMakeClass)

    def MakeDispatchFuncMethod(self, entry, name, bMakeClass=1):
        fdesc = entry.desc
        doc = entry.doc
        names = entry.names
        ret = []
        if bMakeClass:
            linePrefix = "\t"
            defNamedOptArg = "defaultNamedOptArg"
            defNamedNotOptArg = "defaultNamedNotOptArg"
            defUnnamedArg = "defaultUnnamedArg"
        else:
            linePrefix = ""
            defNamedOptArg = "pythoncom.Missing"
            defNamedNotOptArg = "pythoncom.Missing"
            defUnnamedArg = "pythoncom.Missing"
        defOutArg = "pythoncom.Missing"
        id = fdesc[0]

        s = (
            linePrefix
            + "def "
            + name
            + "(self"
            + BuildCallList(
                fdesc,
                names,
                defNamedOptArg,
                defNamedNotOptArg,
                defUnnamedArg,
                defOutArg,
            )
            + "):"
        )
        ret.append(s)
        if doc and doc[1]:
            ret.append(linePrefix + "\t" + _makeDocString(doc[1]))

        resclsid = entry.GetResultCLSID()
        if resclsid:
            resclsid = "'%s'" % resclsid
        else:
            resclsid = "None"
        # Strip the default values from the arg desc
        retDesc = fdesc[8][:2]
        argsDesc = tuple([what[:2] for what in fdesc[2]])
        # The runtime translation of the return types is expensive, so when we know the
        # return type of the function, there is no need to check the type at runtime.
        # To qualify, this function must return a "simple" type, and have no byref args.
        # Check if we have byrefs or anything in the args which mean we still need a translate.
        param_flags = [what[1] for what in fdesc[2]]
        bad_params = [
            flag
            for flag in param_flags
            if flag & (pythoncom.PARAMFLAG_FOUT | pythoncom.PARAMFLAG_FRETVAL) != 0
        ]
        s = None
        if len(bad_params) == 0 and len(retDesc) == 2 and retDesc[1] == 0:
            rd = retDesc[0]
            if rd in NoTranslateMap:
                s = "%s\treturn self._oleobj_.InvokeTypes(%d, LCID, %s, %s, %s%s)" % (
                    linePrefix,
                    id,
                    fdesc[4],
                    retDesc,
                    argsDesc,
                    _BuildArgList(fdesc, names),
                )
            elif rd in [pythoncom.VT_DISPATCH, pythoncom.VT_UNKNOWN]:
                s = "%s\tret = self._oleobj_.InvokeTypes(%d, LCID, %s, %s, %s%s)\n" % (
                    linePrefix,
                    id,
                    fdesc[4],
                    retDesc,
                    repr(argsDesc),
                    _BuildArgList(fdesc, names),
                )
                s = s + "%s\tif ret is not None:\n" % (linePrefix,)
                if rd == pythoncom.VT_UNKNOWN:
                    s = s + "%s\t\t# See if this IUnknown is really an IDispatch\n" % (
                        linePrefix,
                    )
                    s = s + "%s\t\ttry:\n" % (linePrefix,)
                    s = (
                        s
                        + "%s\t\t\tret = ret.QueryInterface(pythoncom.IID_IDispatch)\n"
                        % (linePrefix,)
                    )
                    s = s + "%s\t\texcept pythoncom.error:\n" % (linePrefix,)
                    s = s + "%s\t\t\treturn ret\n" % (linePrefix,)
                s = s + "%s\t\tret = Dispatch(ret, %s, %s)\n" % (
                    linePrefix,
                    repr(name),
                    resclsid,
                )
                s = s + "%s\treturn ret" % (linePrefix)
            elif rd == pythoncom.VT_BSTR:
                s = "%s\t# Result is a Unicode object\n" % (linePrefix,)
                s = (
                    s
                    + "%s\treturn self._oleobj_.InvokeTypes(%d, LCID, %s, %s, %s%s)"
                    % (
                        linePrefix,
                        id,
                        fdesc[4],
                        retDesc,
                        repr(argsDesc),
                        _BuildArgList(fdesc, names),
                    )
                )
            # else s remains None
        if s is None:
            s = "%s\treturn self._ApplyTypes_(%d, %s, %s, %s, %s, %s%s)" % (
                linePrefix,
                id,
                fdesc[4],
                retDesc,
                argsDesc,
                repr(name),
                resclsid,
                _BuildArgList(fdesc, names),
            )

        ret.append(s)
        ret.append("")
        return ret

    def MakeVarArgsFuncMethod(self, entry, name, bMakeClass=1):
        fdesc = entry.desc
        names = entry.names
        doc = entry.doc
        ret = []
        argPrefix = "self"
        if bMakeClass:
            linePrefix = "\t"
        else:
            linePrefix = ""
        ret.append(linePrefix + "def " + name + "(" + argPrefix + ", *args):")
        if doc and doc[1]:
            ret.append(linePrefix + "\t" + _makeDocString(doc[1]))
        if fdesc:
            invoketype = fdesc[4]
        else:
            invoketype = pythoncom.DISPATCH_METHOD
        s = linePrefix + "\treturn self._get_good_object_(self._oleobj_.Invoke(*(("
        ret.append(
            s + str(entry.dispid) + ",0,%d,1)+args)),'%s')" % (invoketype, names[0])
        )
        ret.append("")
        return ret


# Note - "DispatchItem" poorly named - need a new intermediate class.
class VTableItem(DispatchItem):
    def Build(self, typeinfo, attr, bForUser=1):
        DispatchItem.Build(self, typeinfo, attr, bForUser)
        assert typeinfo is not None, "Cant build vtables without type info!"

        meth_list = (
            list(self.mapFuncs.values())
            + list(self.propMapGet.values())
            + list(self.propMapPut.values())
        )
        meth_list.sort(key=lambda m: m.desc[7])

        # Now turn this list into the run-time representation
        # (ready for immediate use or writing to gencache)
        self.vtableFuncs = []
        for entry in meth_list:
            self.vtableFuncs.append((entry.names, entry.dispid, entry.desc))


# A Lazy dispatch item - builds an item on request using info from
# an ITypeComp.  The dynamic module makes the called to build each item,
# and also holds the references to the typeinfo and typecomp.
class LazyDispatchItem(DispatchItem):
    typename = "LazyDispatchItem"

    def __init__(self, attr, doc):
        self.clsid = attr[0]
        DispatchItem.__init__(self, None, attr, doc, 0)


typeSubstMap = {
    pythoncom.VT_INT: pythoncom.VT_I4,
    pythoncom.VT_UINT: pythoncom.VT_UI4,
    pythoncom.VT_HRESULT: pythoncom.VT_I4,
}


def _ResolveType(typerepr, itypeinfo):
    # Resolve VT_USERDEFINED (often aliases or typed IDispatches)

    if type(typerepr) == tuple:
        indir_vt, subrepr = typerepr
        if indir_vt == pythoncom.VT_PTR:
            # If it is a VT_PTR to a VT_USERDEFINED that is an IDispatch/IUnknown,
            # then it resolves to simply the object.
            # Otherwise, it becomes a ByRef of the resolved type
            # We need to drop an indirection level on pointer to user defined interfaces.
            # eg, (VT_PTR, (VT_USERDEFINED, somehandle)) needs to become VT_DISPATCH
            # only when "somehandle" is an object.
            # but (VT_PTR, (VT_USERDEFINED, otherhandle)) doesnt get the indirection dropped.
            was_user = type(subrepr) == tuple and subrepr[0] == pythoncom.VT_USERDEFINED
            subrepr, sub_clsid, sub_doc = _ResolveType(subrepr, itypeinfo)
            if was_user and subrepr in [
                pythoncom.VT_DISPATCH,
                pythoncom.VT_UNKNOWN,
                pythoncom.VT_RECORD,
            ]:
                # Drop the VT_PTR indirection
                return subrepr, sub_clsid, sub_doc
            # Change PTR indirection to byref
            return subrepr | pythoncom.VT_BYREF, sub_clsid, sub_doc
        if indir_vt == pythoncom.VT_SAFEARRAY:
            # resolve the array element, and convert to VT_ARRAY
            subrepr, sub_clsid, sub_doc = _ResolveType(subrepr, itypeinfo)
            return pythoncom.VT_ARRAY | subrepr, sub_clsid, sub_doc
        if indir_vt == pythoncom.VT_CARRAY:  # runtime has no support for this yet.
            # resolve the array element, and convert to VT_CARRAY
            # sheesh - return _something_
            return pythoncom.VT_CARRAY, None, None
        if indir_vt == pythoncom.VT_USERDEFINED:
            try:
                resultTypeInfo = itypeinfo.GetRefTypeInfo(subrepr)
            except pythoncom.com_error as details:
                if details.hresult in [
                    winerror.TYPE_E_CANTLOADLIBRARY,
                    winerror.TYPE_E_LIBNOTREGISTERED,
                ]:
                    # an unregistered interface
                    return pythoncom.VT_UNKNOWN, None, None
                raise

            resultAttr = resultTypeInfo.GetTypeAttr()
            typeKind = resultAttr.typekind
            if typeKind == pythoncom.TKIND_ALIAS:
                tdesc = resultAttr.tdescAlias
                return _ResolveType(tdesc, resultTypeInfo)
            elif typeKind in [pythoncom.TKIND_ENUM, pythoncom.TKIND_MODULE]:
                # For now, assume Long
                return pythoncom.VT_I4, None, None

            elif typeKind == pythoncom.TKIND_DISPATCH:
                clsid = resultTypeInfo.GetTypeAttr()[0]
                retdoc = resultTypeInfo.GetDocumentation(-1)
                return pythoncom.VT_DISPATCH, clsid, retdoc

            elif typeKind in [pythoncom.TKIND_INTERFACE, pythoncom.TKIND_COCLASS]:
                # XXX - should probably get default interface for CO_CLASS???
                clsid = resultTypeInfo.GetTypeAttr()[0]
                retdoc = resultTypeInfo.GetDocumentation(-1)
                return pythoncom.VT_UNKNOWN, clsid, retdoc

            elif typeKind == pythoncom.TKIND_RECORD:
                return pythoncom.VT_RECORD, None, None
            raise NotSupportedException("Can not resolve alias or user-defined type")
    return typeSubstMap.get(typerepr, typerepr), None, None


def _BuildArgList(fdesc, names):
    "Builds list of args to the underlying Invoke method."
    # Word has TypeInfo for Insert() method, but says "no args"
    numArgs = max(fdesc[6], len(fdesc[2]))
    names = list(names)
    while None in names:
        i = names.index(None)
        names[i] = "arg%d" % (i,)
    # We've seen 'source safe' libraries offer the name of 'ret' params in
    # 'names' - although we can't reproduce this, it would be insane to offer
    # more args than we have arg infos for - hence the upper limit on names...
    names = list(map(MakePublicAttributeName, names[1 : (numArgs + 1)]))
    name_num = 0
    while len(names) < numArgs:
        names.append("arg%d" % (len(names),))
    # As per BuildCallList(), avoid huge lines.
    # Hack a "\n" at the end of every 5th name - "strides" would be handy
    # here but don't exist in 2.2
    for i in range(0, len(names), 5):
        names[i] = names[i] + "\n\t\t\t"
    return "," + ", ".join(names)


valid_identifier_chars = string.ascii_letters + string.digits + "_"


def demunge_leading_underscores(className):
    i = 0
    while className[i] == "_":
        i += 1
    assert i >= 2, "Should only be here with names starting with '__'"
    return className[i - 1 :] + className[: i - 1]


# Given a "public name" (eg, the name of a class, function, etc)
# make sure it is a legal (and reasonable!) Python name.
def MakePublicAttributeName(className, is_global=False):
    # Given a class attribute that needs to be public, convert it to a
    # reasonable name.
    # Also need to be careful that the munging doesnt
    # create duplicates - eg, just removing a leading "_" is likely to cause
    # a clash.
    # if is_global is True, then the name is a global variable that may
    # overwrite a builtin - eg, "None"
    if className[:2] == "__":
        return demunge_leading_underscores(className)
    elif className == "None":
        # assign to None is evil (and SyntaxError in 2.4, even though
        # iskeyword says False there) - note that if it was a global
        # it would get picked up below
        className = "NONE"
    elif iskeyword(className):
        # most keywords are lower case (except True, False etc in py3k)
        ret = className.capitalize()
        # but those which aren't get forced upper.
        if ret == className:
            ret = ret.upper()
        return ret
    elif is_global and hasattr(__builtins__, className):
        # builtins may be mixed case.  If capitalizing it doesn't change it,
        # force to all uppercase (eg, "None", "True" become "NONE", "TRUE"
        ret = className.capitalize()
        if ret == className:  # didn't change - force all uppercase.
            ret = ret.upper()
        return ret
    # Strip non printable chars
    return "".join([char for char in className if char in valid_identifier_chars])


# Given a default value passed by a type library, return a string with
# an appropriate repr() for the type.
# Takes a raw ELEMDESC and returns a repr string, or None
# (NOTE: The string itself may be '"None"', which is valid, and different to None.
# XXX - To do: Dates are probably screwed, but can they come in?
def MakeDefaultArgRepr(defArgVal):
    try:
        inOut = defArgVal[1]
    except IndexError:
        # something strange - assume is in param.
        inOut = pythoncom.PARAMFLAG_FIN

    if inOut & pythoncom.PARAMFLAG_FHASDEFAULT:
        # times need special handling...
        val = defArgVal[2]
        if isinstance(val, datetime.datetime):
            # VARIANT <-> SYSTEMTIME conversions always lose any sub-second
            # resolution, so just use a 'timetuple' here.
            return repr(tuple(val.utctimetuple()))
        if type(val) is TimeType:
            # must be the 'old' pywintypes time object...
            year = val.year
            month = val.month
            day = val.day
            hour = val.hour
            minute = val.minute
            second = val.second
            msec = val.msec
            return (
                "pywintypes.Time((%(year)d, %(month)d, %(day)d, %(hour)d, %(minute)d, %(second)d,0,0,0,%(msec)d))"
                % locals()
            )
        return repr(val)
    return None


def BuildCallList(
    fdesc,
    names,
    defNamedOptArg,
    defNamedNotOptArg,
    defUnnamedArg,
    defOutArg,
    is_comment=False,
):
    "Builds a Python declaration for a method."
    # Names[0] is the func name - param names are from 1.
    numArgs = len(fdesc[2])
    numOptArgs = fdesc[6]
    strval = ""
    if numOptArgs == -1:  # Special value that says "var args after here"
        firstOptArg = numArgs
        numArgs = numArgs - 1
    else:
        firstOptArg = numArgs - numOptArgs
    for arg in range(numArgs):
        try:
            argName = names[arg + 1]
            namedArg = argName is not None
        except IndexError:
            namedArg = 0
        if not namedArg:
            argName = "arg%d" % (arg)
        thisdesc = fdesc[2][arg]
        # See if the IDL specified a default value
        defArgVal = MakeDefaultArgRepr(thisdesc)
        if defArgVal is None:
            # Out params always get their special default
            if (
                thisdesc[1] & (pythoncom.PARAMFLAG_FOUT | pythoncom.PARAMFLAG_FIN)
                == pythoncom.PARAMFLAG_FOUT
            ):
                defArgVal = defOutArg
            else:
                # Unnamed arg - always allow default values.
                if namedArg:
                    # Is a named argument
                    if arg >= firstOptArg:
                        defArgVal = defNamedOptArg
                    else:
                        defArgVal = defNamedNotOptArg
                else:
                    defArgVal = defUnnamedArg

        argName = MakePublicAttributeName(argName)
        # insanely long lines with an 'encoding' flag crashes python 2.4.0
        # keep 5 args per line
        # This may still fail if the arg names are insane, but that seems
        # unlikely.  See also _BuildArgList()
        if (arg + 1) % 5 == 0:
            strval = strval + "\n"
            if is_comment:
                strval = strval + "#"
            strval = strval + "\t\t\t"
        strval = strval + ", " + argName
        if defArgVal:
            strval = strval + "=" + defArgVal
    if numOptArgs == -1:
        strval = strval + ", *" + names[-1]

    return strval


if __name__ == "__main__":
    print("Use 'makepy.py' to generate Python code - this module is just a helper")
