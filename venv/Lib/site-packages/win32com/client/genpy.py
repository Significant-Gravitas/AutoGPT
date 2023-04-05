"""genpy.py - The worker for makepy.  See makepy.py for more details

This code was moved simply to speed Python in normal circumstances.  As the makepy.py
is normally run from the command line, it reparses the code each time.  Now makepy
is nothing more than the command line handler and public interface.

The makepy command line etc handling is also getting large enough in its own right!
"""

# NOTE - now supports a "demand" mechanism - the top-level is a package, and
# each class etc can be made individually.
# This should eventually become the default.
# Then the old non-package technique should be removed.
# There should be no b/w compat issues, and will just help clean the code.
# This will be done once the new "demand" mechanism gets a good workout.
import os
import sys
import time

import pythoncom
import win32com

from . import build

error = "makepy.error"
makepy_version = "0.5.01"  # Written to generated file.

GEN_FULL = "full"
GEN_DEMAND_BASE = "demand(base)"
GEN_DEMAND_CHILD = "demand(child)"

# This map is used purely for the users benefit -it shows the
# raw, underlying type of Alias/Enums, etc.  The COM implementation
# does not use this map at runtime - all Alias/Enum have already
# been translated.
mapVTToTypeString = {
    pythoncom.VT_I2: "types.IntType",
    pythoncom.VT_I4: "types.IntType",
    pythoncom.VT_R4: "types.FloatType",
    pythoncom.VT_R8: "types.FloatType",
    pythoncom.VT_BSTR: "types.StringType",
    pythoncom.VT_BOOL: "types.IntType",
    pythoncom.VT_VARIANT: "types.TypeType",
    pythoncom.VT_I1: "types.IntType",
    pythoncom.VT_UI1: "types.IntType",
    pythoncom.VT_UI2: "types.IntType",
    pythoncom.VT_UI4: "types.IntType",
    pythoncom.VT_I8: "types.LongType",
    pythoncom.VT_UI8: "types.LongType",
    pythoncom.VT_INT: "types.IntType",
    pythoncom.VT_DATE: "pythoncom.PyTimeType",
    pythoncom.VT_UINT: "types.IntType",
}


# Given a propget function's arg desc, return the default parameters for all
# params bar the first.  Eg, then Python does a:
# object.Property = "foo"
# Python can only pass the "foo" value.  If the property has
# multiple args, and the rest have default values, this allows
# Python to correctly pass those defaults.
def MakeDefaultArgsForPropertyPut(argsDesc):
    ret = []
    for desc in argsDesc[1:]:
        default = build.MakeDefaultArgRepr(desc)
        if default is None:
            break
        ret.append(default)
    return tuple(ret)


def MakeMapLineEntry(dispid, wFlags, retType, argTypes, user, resultCLSID):
    # Strip the default value
    argTypes = tuple([what[:2] for what in argTypes])
    return '(%s, %d, %s, %s, "%s", %s)' % (
        dispid,
        wFlags,
        retType[:2],
        argTypes,
        user,
        resultCLSID,
    )


def MakeEventMethodName(eventName):
    if eventName[:2] == "On":
        return eventName
    else:
        return "On" + eventName


def WriteSinkEventMap(obj, stream):
    print("\t_dispid_to_func_ = {", file=stream)
    for name, entry in (
        list(obj.propMapGet.items())
        + list(obj.propMapPut.items())
        + list(obj.mapFuncs.items())
    ):
        fdesc = entry.desc
        print(
            '\t\t%9d : "%s",' % (fdesc.memid, MakeEventMethodName(entry.names[0])),
            file=stream,
        )
    print("\t\t}", file=stream)


# MI is used to join my writable helpers, and the OLE
# classes.
class WritableItem:
    # __cmp__ used for sorting in py2x...
    def __cmp__(self, other):
        "Compare for sorting"
        ret = cmp(self.order, other.order)
        if ret == 0 and self.doc:
            ret = cmp(self.doc[0], other.doc[0])
        return ret

    # ... but not used in py3k - __lt__ minimum needed there
    def __lt__(self, other):  # py3k variant
        if self.order == other.order:
            return self.doc < other.doc
        return self.order < other.order

    def __repr__(self):
        return "OleItem: doc=%s, order=%d" % (repr(self.doc), self.order)


class RecordItem(build.OleItem, WritableItem):
    order = 9
    typename = "RECORD"

    def __init__(self, typeInfo, typeAttr, doc=None, bForUser=1):
        ##    sys.stderr.write("Record %s: size %s\n" % (doc,typeAttr.cbSizeInstance))
        ##    sys.stderr.write(" cVars = %s\n" % (typeAttr.cVars,))
        ##    for i in range(typeAttr.cVars):
        ##        vdesc = typeInfo.GetVarDesc(i)
        ##        sys.stderr.write(" Var %d has value %s, type %d, desc=%s\n" % (i, vdesc.value, vdesc.varkind, vdesc.elemdescVar))
        ##        sys.stderr.write(" Doc is %s\n" % (typeInfo.GetDocumentation(vdesc.memid),))

        build.OleItem.__init__(self, doc)
        self.clsid = typeAttr[0]

    def WriteClass(self, generator):
        pass


# Given an enum, write all aliases for it.
# (no longer necessary for new style code, but still used for old code.
def WriteAliasesForItem(item, aliasItems, stream):
    for alias in aliasItems.values():
        if item.doc and alias.aliasDoc and (alias.aliasDoc[0] == item.doc[0]):
            alias.WriteAliasItem(aliasItems, stream)


class AliasItem(build.OleItem, WritableItem):
    order = 2
    typename = "ALIAS"

    def __init__(self, typeinfo, attr, doc=None, bForUser=1):
        build.OleItem.__init__(self, doc)

        ai = attr[14]
        self.attr = attr
        if type(ai) == type(()) and type(ai[1]) == type(
            0
        ):  # XXX - This is a hack - why tuples?  Need to resolve?
            href = ai[1]
            alinfo = typeinfo.GetRefTypeInfo(href)
            self.aliasDoc = alinfo.GetDocumentation(-1)
            self.aliasAttr = alinfo.GetTypeAttr()
        else:
            self.aliasDoc = None
            self.aliasAttr = None

    def WriteAliasItem(self, aliasDict, stream):
        # we could have been written as part of an alias dependency
        if self.bWritten:
            return

        if self.aliasDoc:
            depName = self.aliasDoc[0]
            if depName in aliasDict:
                aliasDict[depName].WriteAliasItem(aliasDict, stream)
            print(self.doc[0] + " = " + depName, file=stream)
        else:
            ai = self.attr[14]
            if type(ai) == type(0):
                try:
                    typeStr = mapVTToTypeString[ai]
                    print("# %s=%s" % (self.doc[0], typeStr), file=stream)
                except KeyError:
                    print(
                        self.doc[0] + " = None # Can't convert alias info " + str(ai),
                        file=stream,
                    )
        print(file=stream)
        self.bWritten = 1


class EnumerationItem(build.OleItem, WritableItem):
    order = 1
    typename = "ENUMERATION"

    def __init__(self, typeinfo, attr, doc=None, bForUser=1):
        build.OleItem.__init__(self, doc)

        self.clsid = attr[0]
        self.mapVars = {}
        typeFlags = attr[11]
        self.hidden = (
            typeFlags & pythoncom.TYPEFLAG_FHIDDEN
            or typeFlags & pythoncom.TYPEFLAG_FRESTRICTED
        )

        for j in range(attr[7]):
            vdesc = typeinfo.GetVarDesc(j)
            name = typeinfo.GetNames(vdesc[0])[0]
            self.mapVars[name] = build.MapEntry(vdesc)

    ##  def WriteEnumerationHeaders(self, aliasItems, stream):
    ##    enumName = self.doc[0]
    ##    print >> stream "%s=constants # Compatibility with previous versions." % (enumName)
    ##    WriteAliasesForItem(self, aliasItems)

    def WriteEnumerationItems(self, stream):
        num = 0
        enumName = self.doc[0]
        # Write in name alpha order
        names = list(self.mapVars.keys())
        names.sort()
        for name in names:
            entry = self.mapVars[name]
            vdesc = entry.desc
            if vdesc[4] == pythoncom.VAR_CONST:
                val = vdesc[1]

                use = repr(val)
                # Make sure the repr of the value is valid python syntax
                # still could cause an error on import if it contains a module or type name
                # not available in the global namespace
                try:
                    compile(use, "<makepy>", "eval")
                except SyntaxError:
                    # At least add the repr as a string, so it can be investigated further
                    # Sanitize it, in case the repr contains its own quotes.  (??? line breaks too ???)
                    use = use.replace('"', "'")
                    use = (
                        '"'
                        + use
                        + '"'
                        + " # This VARIANT type cannot be converted automatically"
                    )
                print(
                    "\t%-30s=%-10s # from enum %s"
                    % (build.MakePublicAttributeName(name, True), use, enumName),
                    file=stream,
                )
                num += 1
        return num


class VTableItem(build.VTableItem, WritableItem):
    order = 4

    def WriteClass(self, generator):
        self.WriteVTableMap(generator)
        self.bWritten = 1

    def WriteVTableMap(self, generator):
        stream = generator.file
        print(
            "%s_vtables_dispatch_ = %d" % (self.python_name, self.bIsDispatch),
            file=stream,
        )
        print("%s_vtables_ = [" % (self.python_name,), file=stream)
        for v in self.vtableFuncs:
            names, dispid, desc = v
            assert desc.desckind == pythoncom.DESCKIND_FUNCDESC
            arg_reprs = []
            # more hoops so we don't generate huge lines.
            item_num = 0
            print("\t((", end=" ", file=stream)
            for name in names:
                print(repr(name), ",", end=" ", file=stream)
                item_num = item_num + 1
                if item_num % 5 == 0:
                    print("\n\t\t\t", end=" ", file=stream)
            print(
                "), %d, (%r, %r, [" % (dispid, desc.memid, desc.scodeArray),
                end=" ",
                file=stream,
            )
            for arg in desc.args:
                item_num = item_num + 1
                if item_num % 5 == 0:
                    print("\n\t\t\t", end=" ", file=stream)
                defval = build.MakeDefaultArgRepr(arg)
                if arg[3] is None:
                    arg3_repr = None
                else:
                    arg3_repr = repr(arg[3])
                print(
                    repr((arg[0], arg[1], defval, arg3_repr)), ",", end=" ", file=stream
                )
            print("],", end=" ", file=stream)
            print(repr(desc.funckind), ",", end=" ", file=stream)
            print(repr(desc.invkind), ",", end=" ", file=stream)
            print(repr(desc.callconv), ",", end=" ", file=stream)
            print(repr(desc.cParamsOpt), ",", end=" ", file=stream)
            print(repr(desc.oVft), ",", end=" ", file=stream)
            print(repr(desc.rettype), ",", end=" ", file=stream)
            print(repr(desc.wFuncFlags), ",", end=" ", file=stream)
            print(")),", file=stream)
        print("]", file=stream)
        print(file=stream)


class DispatchItem(build.DispatchItem, WritableItem):
    order = 3

    def __init__(self, typeinfo, attr, doc=None):
        build.DispatchItem.__init__(self, typeinfo, attr, doc)
        self.type_attr = attr
        self.coclass_clsid = None

    def WriteClass(self, generator):
        if (
            not self.bIsDispatch
            and not self.type_attr.typekind == pythoncom.TKIND_DISPATCH
        ):
            return
        # This is pretty screwey - now we have vtable support we
        # should probably rethink this (ie, maybe write both sides for sinks, etc)
        if self.bIsSink:
            self.WriteEventSinkClassHeader(generator)
            self.WriteCallbackClassBody(generator)
        else:
            self.WriteClassHeader(generator)
            self.WriteClassBody(generator)
        print(file=generator.file)
        self.bWritten = 1

    def WriteClassHeader(self, generator):
        generator.checkWriteDispatchBaseClass()
        doc = self.doc
        stream = generator.file
        print("class " + self.python_name + "(DispatchBaseClass):", file=stream)
        if doc[1]:
            print("\t" + build._makeDocString(doc[1]), file=stream)
        try:
            progId = pythoncom.ProgIDFromCLSID(self.clsid)
            print(
                "\t# This class is creatable by the name '%s'" % (progId), file=stream
            )
        except pythoncom.com_error:
            pass
        print("\tCLSID = " + repr(self.clsid), file=stream)
        if self.coclass_clsid is None:
            print("\tcoclass_clsid = None", file=stream)
        else:
            print("\tcoclass_clsid = " + repr(self.coclass_clsid), file=stream)
        print(file=stream)
        self.bWritten = 1

    def WriteEventSinkClassHeader(self, generator):
        generator.checkWriteEventBaseClass()
        doc = self.doc
        stream = generator.file
        print("class " + self.python_name + ":", file=stream)
        if doc[1]:
            print("\t" + build._makeDocString(doc[1]), file=stream)
        try:
            progId = pythoncom.ProgIDFromCLSID(self.clsid)
            print(
                "\t# This class is creatable by the name '%s'" % (progId), file=stream
            )
        except pythoncom.com_error:
            pass
        print("\tCLSID = CLSID_Sink = " + repr(self.clsid), file=stream)
        if self.coclass_clsid is None:
            print("\tcoclass_clsid = None", file=stream)
        else:
            print("\tcoclass_clsid = " + repr(self.coclass_clsid), file=stream)
        print("\t_public_methods_ = [] # For COM Server support", file=stream)
        WriteSinkEventMap(self, stream)
        print(file=stream)
        print("\tdef __init__(self, oobj = None):", file=stream)
        print("\t\tif oobj is None:", file=stream)
        print("\t\t\tself._olecp = None", file=stream)
        print("\t\telse:", file=stream)
        print("\t\t\timport win32com.server.util", file=stream)
        print(
            "\t\t\tfrom win32com.server.policy import EventHandlerPolicy", file=stream
        )
        print(
            "\t\t\tcpc=oobj._oleobj_.QueryInterface(pythoncom.IID_IConnectionPointContainer)",
            file=stream,
        )
        print("\t\t\tcp=cpc.FindConnectionPoint(self.CLSID_Sink)", file=stream)
        print(
            "\t\t\tcookie=cp.Advise(win32com.server.util.wrap(self, usePolicy=EventHandlerPolicy))",
            file=stream,
        )
        print("\t\t\tself._olecp,self._olecp_cookie = cp,cookie", file=stream)
        print("\tdef __del__(self):", file=stream)
        print("\t\ttry:", file=stream)
        print("\t\t\tself.close()", file=stream)
        print("\t\texcept pythoncom.com_error:", file=stream)
        print("\t\t\tpass", file=stream)
        print("\tdef close(self):", file=stream)
        print("\t\tif self._olecp is not None:", file=stream)
        print(
            "\t\t\tcp,cookie,self._olecp,self._olecp_cookie = self._olecp,self._olecp_cookie,None,None",
            file=stream,
        )
        print("\t\t\tcp.Unadvise(cookie)", file=stream)
        print("\tdef _query_interface_(self, iid):", file=stream)
        print("\t\timport win32com.server.util", file=stream)
        print(
            "\t\tif iid==self.CLSID_Sink: return win32com.server.util.wrap(self)",
            file=stream,
        )
        print(file=stream)
        self.bWritten = 1

    def WriteCallbackClassBody(self, generator):
        stream = generator.file
        print("\t# Event Handlers", file=stream)
        print(
            "\t# If you create handlers, they should have the following prototypes:",
            file=stream,
        )
        for name, entry in (
            list(self.propMapGet.items())
            + list(self.propMapPut.items())
            + list(self.mapFuncs.items())
        ):
            fdesc = entry.desc
            methName = MakeEventMethodName(entry.names[0])
            print(
                "#\tdef "
                + methName
                + "(self"
                + build.BuildCallList(
                    fdesc,
                    entry.names,
                    "defaultNamedOptArg",
                    "defaultNamedNotOptArg",
                    "defaultUnnamedArg",
                    "pythoncom.Missing",
                    is_comment=True,
                )
                + "):",
                file=stream,
            )
            if entry.doc and entry.doc[1]:
                print("#\t\t" + build._makeDocString(entry.doc[1]), file=stream)
        print(file=stream)
        self.bWritten = 1

    def WriteClassBody(self, generator):
        stream = generator.file
        # Write in alpha order.
        names = list(self.mapFuncs.keys())
        names.sort()
        specialItems = {
            "count": None,
            "item": None,
            "value": None,
            "_newenum": None,
        }  # If found, will end up with (entry, invoke_tupe)
        itemCount = None
        for name in names:
            entry = self.mapFuncs[name]
            assert entry.desc.desckind == pythoncom.DESCKIND_FUNCDESC
            # skip [restricted] methods, unless it is the
            # enumerator (which, being part of the "system",
            # we know about and can use)
            dispid = entry.desc.memid
            if (
                entry.desc.wFuncFlags & pythoncom.FUNCFLAG_FRESTRICTED
                and dispid != pythoncom.DISPID_NEWENUM
            ):
                continue
            # If not accessible via IDispatch, then we can't use it here.
            if entry.desc.funckind != pythoncom.FUNC_DISPATCH:
                continue
            if dispid == pythoncom.DISPID_VALUE:
                lkey = "value"
            elif dispid == pythoncom.DISPID_NEWENUM:
                specialItems["_newenum"] = (entry, entry.desc.invkind, None)
                continue  # Dont build this one now!
            else:
                lkey = name.lower()
            if (
                lkey in specialItems and specialItems[lkey] is None
            ):  # remember if a special one.
                specialItems[lkey] = (entry, entry.desc.invkind, None)
            if generator.bBuildHidden or not entry.hidden:
                if entry.GetResultName():
                    print("\t# Result is of type " + entry.GetResultName(), file=stream)
                if entry.wasProperty:
                    print(
                        "\t# The method %s is actually a property, but must be used as a method to correctly pass the arguments"
                        % name,
                        file=stream,
                    )
                ret = self.MakeFuncMethod(entry, build.MakePublicAttributeName(name))
                for line in ret:
                    print(line, file=stream)
        print("\t_prop_map_get_ = {", file=stream)
        names = list(self.propMap.keys())
        names.sort()
        for key in names:
            entry = self.propMap[key]
            if generator.bBuildHidden or not entry.hidden:
                resultName = entry.GetResultName()
                if resultName:
                    print(
                        "\t\t# Property '%s' is an object of type '%s'"
                        % (key, resultName),
                        file=stream,
                    )
                lkey = key.lower()
                details = entry.desc
                resultDesc = details[2]
                argDesc = ()
                mapEntry = MakeMapLineEntry(
                    details.memid,
                    pythoncom.DISPATCH_PROPERTYGET,
                    resultDesc,
                    argDesc,
                    key,
                    entry.GetResultCLSIDStr(),
                )

                if details.memid == pythoncom.DISPID_VALUE:
                    lkey = "value"
                elif details.memid == pythoncom.DISPID_NEWENUM:
                    lkey = "_newenum"
                else:
                    lkey = key.lower()
                if (
                    lkey in specialItems and specialItems[lkey] is None
                ):  # remember if a special one.
                    specialItems[lkey] = (
                        entry,
                        pythoncom.DISPATCH_PROPERTYGET,
                        mapEntry,
                    )
                    # All special methods, except _newenum, are written
                    # "normally".  This is a mess!
                    if details.memid == pythoncom.DISPID_NEWENUM:
                        continue

                print(
                    '\t\t"%s": %s,' % (build.MakePublicAttributeName(key), mapEntry),
                    file=stream,
                )
        names = list(self.propMapGet.keys())
        names.sort()
        for key in names:
            entry = self.propMapGet[key]
            if generator.bBuildHidden or not entry.hidden:
                if entry.GetResultName():
                    print(
                        "\t\t# Method '%s' returns object of type '%s'"
                        % (key, entry.GetResultName()),
                        file=stream,
                    )
                details = entry.desc
                assert details.desckind == pythoncom.DESCKIND_FUNCDESC
                lkey = key.lower()
                argDesc = details[2]
                resultDesc = details[8]
                mapEntry = MakeMapLineEntry(
                    details[0],
                    pythoncom.DISPATCH_PROPERTYGET,
                    resultDesc,
                    argDesc,
                    key,
                    entry.GetResultCLSIDStr(),
                )
                if details.memid == pythoncom.DISPID_VALUE:
                    lkey = "value"
                elif details.memid == pythoncom.DISPID_NEWENUM:
                    lkey = "_newenum"
                else:
                    lkey = key.lower()
                if (
                    lkey in specialItems and specialItems[lkey] is None
                ):  # remember if a special one.
                    specialItems[lkey] = (
                        entry,
                        pythoncom.DISPATCH_PROPERTYGET,
                        mapEntry,
                    )
                    # All special methods, except _newenum, are written
                    # "normally".  This is a mess!
                    if details.memid == pythoncom.DISPID_NEWENUM:
                        continue
                print(
                    '\t\t"%s": %s,' % (build.MakePublicAttributeName(key), mapEntry),
                    file=stream,
                )

        print("\t}", file=stream)

        print("\t_prop_map_put_ = {", file=stream)
        # These are "Invoke" args
        names = list(self.propMap.keys())
        names.sort()
        for key in names:
            entry = self.propMap[key]
            if generator.bBuildHidden or not entry.hidden:
                lkey = key.lower()
                details = entry.desc
                # If default arg is None, write an empty tuple
                defArgDesc = build.MakeDefaultArgRepr(details[2])
                if defArgDesc is None:
                    defArgDesc = ""
                else:
                    defArgDesc = defArgDesc + ","
                print(
                    '\t\t"%s" : ((%s, LCID, %d, 0),(%s)),'
                    % (
                        build.MakePublicAttributeName(key),
                        details[0],
                        pythoncom.DISPATCH_PROPERTYPUT,
                        defArgDesc,
                    ),
                    file=stream,
                )

        names = list(self.propMapPut.keys())
        names.sort()
        for key in names:
            entry = self.propMapPut[key]
            if generator.bBuildHidden or not entry.hidden:
                details = entry.desc
                defArgDesc = MakeDefaultArgsForPropertyPut(details[2])
                print(
                    '\t\t"%s": ((%s, LCID, %d, 0),%s),'
                    % (
                        build.MakePublicAttributeName(key),
                        details[0],
                        details[4],
                        defArgDesc,
                    ),
                    file=stream,
                )
        print("\t}", file=stream)

        if specialItems["value"]:
            entry, invoketype, propArgs = specialItems["value"]
            if propArgs is None:
                typename = "method"
                ret = self.MakeFuncMethod(entry, "__call__")
            else:
                typename = "property"
                ret = [
                    "\tdef __call__(self):\n\t\treturn self._ApplyTypes_(*%s)"
                    % propArgs
                ]
            print(
                "\t# Default %s for this class is '%s'" % (typename, entry.names[0]),
                file=stream,
            )
            for line in ret:
                print(line, file=stream)
            print("\tdef __str__(self, *args):", file=stream)
            print("\t\treturn str(self.__call__(*args))", file=stream)
            print("\tdef __int__(self, *args):", file=stream)
            print("\t\treturn int(self.__call__(*args))", file=stream)

        # _NewEnum (DISPID_NEWENUM) does not appear in typelib for many office objects,
        # but it can still be retrieved at runtime, so  always create __iter__.
        # Also, some of those same objects use 1-based indexing, causing the old-style
        # __getitem__ iteration to fail for index 0 where the dynamic iteration succeeds.
        if specialItems["_newenum"]:
            enumEntry, invoketype, propArgs = specialItems["_newenum"]
            assert enumEntry.desc.desckind == pythoncom.DESCKIND_FUNCDESC
            invkind = enumEntry.desc.invkind
            # ??? Wouldn't this be the resultCLSID for the iterator itself, rather than the resultCLSID
            #  for the result of each Next() call, which is what it's used for ???
            resultCLSID = enumEntry.GetResultCLSIDStr()
        else:
            invkind = pythoncom.DISPATCH_METHOD | pythoncom.DISPATCH_PROPERTYGET
            resultCLSID = "None"
        # If we dont have a good CLSID for the enum result, assume it is the same as the Item() method.
        if resultCLSID == "None" and "Item" in self.mapFuncs:
            resultCLSID = self.mapFuncs["Item"].GetResultCLSIDStr()
        print("\tdef __iter__(self):", file=stream)
        print('\t\t"Return a Python iterator for this object"', file=stream)
        print("\t\ttry:", file=stream)
        print(
            "\t\t\tob = self._oleobj_.InvokeTypes(%d,LCID,%d,(13, 10),())"
            % (pythoncom.DISPID_NEWENUM, invkind),
            file=stream,
        )
        print("\t\texcept pythoncom.error:", file=stream)
        print(
            '\t\t\traise TypeError("This object does not support enumeration")',
            file=stream,
        )
        # Iterator is wrapped as PyIEnumVariant, and each result of __next__ is Dispatch'ed if necessary
        print(
            "\t\treturn win32com.client.util.Iterator(ob, %s)" % resultCLSID,
            file=stream,
        )

        if specialItems["item"]:
            entry, invoketype, propArgs = specialItems["item"]
            resultCLSID = entry.GetResultCLSIDStr()
            print(
                "\t#This class has Item property/method which allows indexed access with the object[key] syntax.",
                file=stream,
            )
            print(
                "\t#Some objects will accept a string or other type of key in addition to integers.",
                file=stream,
            )
            print(
                "\t#Note that many Office objects do not use zero-based indexing.",
                file=stream,
            )
            print("\tdef __getitem__(self, key):", file=stream)
            print(
                '\t\treturn self._get_good_object_(self._oleobj_.Invoke(*(%d, LCID, %d, 1, key)), "Item", %s)'
                % (entry.desc.memid, invoketype, resultCLSID),
                file=stream,
            )

        if specialItems["count"]:
            entry, invoketype, propArgs = specialItems["count"]
            if propArgs is None:
                typename = "method"
                ret = self.MakeFuncMethod(entry, "__len__")
            else:
                typename = "property"
                ret = [
                    "\tdef __len__(self):\n\t\treturn self._ApplyTypes_(*%s)" % propArgs
                ]
            print(
                "\t#This class has Count() %s - allow len(ob) to provide this"
                % (typename),
                file=stream,
            )
            for line in ret:
                print(line, file=stream)
            # Also include a __nonzero__
            print(
                "\t#This class has a __len__ - this is needed so 'if object:' always returns TRUE.",
                file=stream,
            )
            print("\tdef __nonzero__(self):", file=stream)
            print("\t\treturn True", file=stream)


class CoClassItem(build.OleItem, WritableItem):
    order = 5
    typename = "COCLASS"

    def __init__(self, typeinfo, attr, doc=None, sources=[], interfaces=[], bForUser=1):
        build.OleItem.__init__(self, doc)
        self.clsid = attr[0]
        self.sources = sources
        self.interfaces = interfaces
        self.bIsDispatch = 1  # Pretend it is so it is written to the class map.

    def WriteClass(self, generator):
        generator.checkWriteCoClassBaseClass()
        doc = self.doc
        stream = generator.file
        if generator.generate_type == GEN_DEMAND_CHILD:
            # Some special imports we must setup.
            referenced_items = []
            for ref, flag in self.sources:
                referenced_items.append(ref)
            for ref, flag in self.interfaces:
                referenced_items.append(ref)
            print("import sys", file=stream)
            for ref in referenced_items:
                print(
                    "__import__('%s.%s')" % (generator.base_mod_name, ref.python_name),
                    file=stream,
                )
                print(
                    "%s = sys.modules['%s.%s'].%s"
                    % (
                        ref.python_name,
                        generator.base_mod_name,
                        ref.python_name,
                        ref.python_name,
                    ),
                    file=stream,
                )
                # And pretend we have written it - the name is now available as if we had!
                ref.bWritten = 1
        try:
            progId = pythoncom.ProgIDFromCLSID(self.clsid)
            print("# This CoClass is known by the name '%s'" % (progId), file=stream)
        except pythoncom.com_error:
            pass
        print(
            "class %s(CoClassBaseClass): # A CoClass" % (self.python_name), file=stream
        )
        if doc and doc[1]:
            print("\t# " + doc[1], file=stream)
        print("\tCLSID = %r" % (self.clsid,), file=stream)
        print("\tcoclass_sources = [", file=stream)
        defItem = None
        for item, flag in self.sources:
            if flag & pythoncom.IMPLTYPEFLAG_FDEFAULT:
                defItem = item
            # If we have written a Python class, reference the name -
            # otherwise just the IID.
            if item.bWritten:
                key = item.python_name
            else:
                key = repr(str(item.clsid))  # really the iid.
            print("\t\t%s," % (key), file=stream)
        print("\t]", file=stream)
        if defItem:
            if defItem.bWritten:
                defName = defItem.python_name
            else:
                defName = repr(str(defItem.clsid))  # really the iid.
            print("\tdefault_source = %s" % (defName,), file=stream)
        print("\tcoclass_interfaces = [", file=stream)
        defItem = None
        for item, flag in self.interfaces:
            if flag & pythoncom.IMPLTYPEFLAG_FDEFAULT:  # and dual:
                defItem = item
            # If we have written a class, reference its name, otherwise the IID
            if item.bWritten:
                key = item.python_name
            else:
                key = repr(str(item.clsid))  # really the iid.
            print("\t\t%s," % (key,), file=stream)
        print("\t]", file=stream)
        if defItem:
            if defItem.bWritten:
                defName = defItem.python_name
            else:
                defName = repr(str(defItem.clsid))  # really the iid.
            print("\tdefault_interface = %s" % (defName,), file=stream)
        self.bWritten = 1
        print(file=stream)


class GeneratorProgress:
    def __init__(self):
        pass

    def Starting(self, tlb_desc):
        """Called when the process starts."""
        self.tlb_desc = tlb_desc

    def Finished(self):
        """Called when the process is complete."""

    def SetDescription(self, desc, maxticks=None):
        """We are entering a major step.  If maxticks, then this
        is how many ticks we expect to make until finished
        """

    def Tick(self, desc=None):
        """Minor progress step.  Can provide new description if necessary"""

    def VerboseProgress(self, desc):
        """Verbose/Debugging output."""

    def LogWarning(self, desc):
        """If a warning is generated"""

    def LogBeginGenerate(self, filename):
        pass

    def Close(self):
        pass


class Generator:
    def __init__(
        self,
        typelib,
        sourceFilename,
        progressObject,
        bBuildHidden=1,
        bUnicodeToString=None,
    ):
        assert bUnicodeToString is None, "this is deprecated and will go away"
        self.bHaveWrittenDispatchBaseClass = 0
        self.bHaveWrittenCoClassBaseClass = 0
        self.bHaveWrittenEventBaseClass = 0
        self.typelib = typelib
        self.sourceFilename = sourceFilename
        self.bBuildHidden = bBuildHidden
        self.progress = progressObject
        # These 2 are later additions and most of the code still 'print's...
        self.file = None

    def CollectOleItemInfosFromType(self):
        ret = []
        for i in range(self.typelib.GetTypeInfoCount()):
            info = self.typelib.GetTypeInfo(i)
            infotype = self.typelib.GetTypeInfoType(i)
            doc = self.typelib.GetDocumentation(i)
            attr = info.GetTypeAttr()
            ret.append((info, infotype, doc, attr))
        return ret

    def _Build_CoClass(self, type_info_tuple):
        info, infotype, doc, attr = type_info_tuple
        # find the source and dispinterfaces for the coclass
        child_infos = []
        for j in range(attr[8]):
            flags = info.GetImplTypeFlags(j)
            try:
                refType = info.GetRefTypeInfo(info.GetRefTypeOfImplType(j))
            except pythoncom.com_error:
                # Can't load a dependent typelib?
                continue
            refAttr = refType.GetTypeAttr()
            child_infos.append(
                (
                    info,
                    refAttr.typekind,
                    refType,
                    refType.GetDocumentation(-1),
                    refAttr,
                    flags,
                )
            )

        # Done generating children - now the CoClass itself.
        newItem = CoClassItem(info, attr, doc)
        return newItem, child_infos

    def _Build_CoClassChildren(self, coclass, coclass_info, oleItems, vtableItems):
        sources = {}
        interfaces = {}
        for info, info_type, refType, doc, refAttr, flags in coclass_info:
            #          sys.stderr.write("Attr typeflags for coclass referenced object %s=%d (%d), typekind=%d\n" % (name, refAttr.wTypeFlags, refAttr.wTypeFlags & pythoncom.TYPEFLAG_FDUAL,refAttr.typekind))
            if refAttr.typekind == pythoncom.TKIND_DISPATCH or (
                refAttr.typekind == pythoncom.TKIND_INTERFACE
                and refAttr[11] & pythoncom.TYPEFLAG_FDISPATCHABLE
            ):
                clsid = refAttr[0]
                if clsid in oleItems:
                    dispItem = oleItems[clsid]
                else:
                    dispItem = DispatchItem(refType, refAttr, doc)
                    oleItems[dispItem.clsid] = dispItem
                dispItem.coclass_clsid = coclass.clsid
                if flags & pythoncom.IMPLTYPEFLAG_FSOURCE:
                    dispItem.bIsSink = 1
                    sources[dispItem.clsid] = (dispItem, flags)
                else:
                    interfaces[dispItem.clsid] = (dispItem, flags)
                # If dual interface, make do that too.
                if clsid not in vtableItems and refAttr[11] & pythoncom.TYPEFLAG_FDUAL:
                    refType = refType.GetRefTypeInfo(refType.GetRefTypeOfImplType(-1))
                    refAttr = refType.GetTypeAttr()
                    assert (
                        refAttr.typekind == pythoncom.TKIND_INTERFACE
                    ), "must be interface bynow!"
                    vtableItem = VTableItem(refType, refAttr, doc)
                    vtableItems[clsid] = vtableItem
        coclass.sources = list(sources.values())
        coclass.interfaces = list(interfaces.values())

    def _Build_Interface(self, type_info_tuple):
        info, infotype, doc, attr = type_info_tuple
        oleItem = vtableItem = None
        if infotype == pythoncom.TKIND_DISPATCH or (
            infotype == pythoncom.TKIND_INTERFACE
            and attr[11] & pythoncom.TYPEFLAG_FDISPATCHABLE
        ):
            oleItem = DispatchItem(info, attr, doc)
            # If this DISPATCH interface dual, then build that too.
            if attr.wTypeFlags & pythoncom.TYPEFLAG_FDUAL:
                # Get the vtable interface
                refhtype = info.GetRefTypeOfImplType(-1)
                info = info.GetRefTypeInfo(refhtype)
                attr = info.GetTypeAttr()
                infotype = pythoncom.TKIND_INTERFACE
            else:
                infotype = None
        assert infotype in [
            None,
            pythoncom.TKIND_INTERFACE,
        ], "Must be a real interface at this point"
        if infotype == pythoncom.TKIND_INTERFACE:
            vtableItem = VTableItem(info, attr, doc)
        return oleItem, vtableItem

    def BuildOleItemsFromType(self):
        assert (
            self.bBuildHidden
        ), "This code doesnt look at the hidden flag - I thought everyone set it true!?!?!"
        oleItems = {}
        enumItems = {}
        recordItems = {}
        vtableItems = {}

        for type_info_tuple in self.CollectOleItemInfosFromType():
            info, infotype, doc, attr = type_info_tuple
            clsid = attr[0]
            if infotype == pythoncom.TKIND_ENUM or infotype == pythoncom.TKIND_MODULE:
                newItem = EnumerationItem(info, attr, doc)
                enumItems[newItem.doc[0]] = newItem
            # We never hide interfaces (MSAccess, for example, nominates interfaces as
            # hidden, assuming that you only ever use them via the CoClass)
            elif infotype in [pythoncom.TKIND_DISPATCH, pythoncom.TKIND_INTERFACE]:
                if clsid not in oleItems:
                    oleItem, vtableItem = self._Build_Interface(type_info_tuple)
                    oleItems[clsid] = oleItem  # Even "None" goes in here.
                    if vtableItem is not None:
                        vtableItems[clsid] = vtableItem
            elif (
                infotype == pythoncom.TKIND_RECORD or infotype == pythoncom.TKIND_UNION
            ):
                newItem = RecordItem(info, attr, doc)
                recordItems[newItem.clsid] = newItem
            elif infotype == pythoncom.TKIND_ALIAS:
                # We dont care about alias' - handled intrinsicly.
                continue
            elif infotype == pythoncom.TKIND_COCLASS:
                newItem, child_infos = self._Build_CoClass(type_info_tuple)
                self._Build_CoClassChildren(newItem, child_infos, oleItems, vtableItems)
                oleItems[newItem.clsid] = newItem
            else:
                self.progress.LogWarning("Unknown TKIND found: %d" % infotype)

        return oleItems, enumItems, recordItems, vtableItems

    def open_writer(self, filename, encoding="mbcs"):
        # A place to put code to open a file with the appropriate encoding.
        # Does *not* set self.file - just opens and returns a file.
        # Actually returns a handle to a temp file - finish_writer then deletes
        # the filename asked for and puts everything back in place.  This
        # is so errors don't leave a 1/2 generated file around causing bizarre
        # errors later, and so that multiple processes writing the same file
        # don't step on each others' toes.
        # Could be a classmethod one day...
        temp_filename = self.get_temp_filename(filename)
        return open(temp_filename, "wt", encoding=encoding)

    def finish_writer(self, filename, f, worked):
        f.close()
        try:
            os.unlink(filename)
        except os.error:
            pass
        temp_filename = self.get_temp_filename(filename)
        if worked:
            try:
                os.rename(temp_filename, filename)
            except os.error:
                # If we are really unlucky, another process may have written the
                # file in between our calls to os.unlink and os.rename. So try
                # again, but only once.
                # There are still some race conditions, but they seem difficult to
                # fix, and they probably occur much less frequently:
                # * The os.rename failure could occur more than once if more than
                #   two processes are involved.
                # * In between os.unlink and os.rename, another process could try
                #   to import the module, having seen that it already exists.
                # * If another process starts a COM server while we are still
                #   generating __init__.py, that process sees that the folder
                #   already exists and assumes that __init__.py is already there
                #   as well.
                try:
                    os.unlink(filename)
                except os.error:
                    pass
                os.rename(temp_filename, filename)
        else:
            os.unlink(temp_filename)

    def get_temp_filename(self, filename):
        return "%s.%d.temp" % (filename, os.getpid())

    def generate(self, file, is_for_demand=0):
        if is_for_demand:
            self.generate_type = GEN_DEMAND_BASE
        else:
            self.generate_type = GEN_FULL
        self.file = file
        self.do_generate()
        self.file = None
        self.progress.Finished()

    def do_gen_file_header(self):
        la = self.typelib.GetLibAttr()
        moduleDoc = self.typelib.GetDocumentation(-1)
        docDesc = ""
        if moduleDoc[1]:
            docDesc = moduleDoc[1]

        # Reset all the 'per file' state
        self.bHaveWrittenDispatchBaseClass = 0
        self.bHaveWrittenCoClassBaseClass = 0
        self.bHaveWrittenEventBaseClass = 0
        # You must provide a file correctly configured for writing unicode.
        # We assert this is it may indicate somewhere in pywin32 that needs
        # upgrading.
        assert self.file.encoding, self.file
        encoding = self.file.encoding  # or "mbcs"

        print("# -*- coding: %s -*-" % (encoding,), file=self.file)
        print("# Created by makepy.py version %s" % (makepy_version,), file=self.file)
        print(
            "# By python version %s" % (sys.version.replace("\n", "-"),), file=self.file
        )
        if self.sourceFilename:
            print(
                "# From type library '%s'" % (os.path.split(self.sourceFilename)[1],),
                file=self.file,
            )
        print("# On %s" % time.ctime(time.time()), file=self.file)

        print(build._makeDocString(docDesc), file=self.file)

        print("makepy_version =", repr(makepy_version), file=self.file)
        print("python_version = 0x%x" % (sys.hexversion,), file=self.file)
        print(file=self.file)
        print(
            "import win32com.client.CLSIDToClass, pythoncom, pywintypes", file=self.file
        )
        print("import win32com.client.util", file=self.file)
        print("from pywintypes import IID", file=self.file)
        print("from win32com.client import Dispatch", file=self.file)
        print(file=self.file)
        print(
            "# The following 3 lines may need tweaking for the particular server",
            file=self.file,
        )
        print(
            "# Candidates are pythoncom.Missing, .Empty and .ArgNotFound",
            file=self.file,
        )
        print("defaultNamedOptArg=pythoncom.Empty", file=self.file)
        print("defaultNamedNotOptArg=pythoncom.Empty", file=self.file)
        print("defaultUnnamedArg=pythoncom.Empty", file=self.file)
        print(file=self.file)
        print("CLSID = " + repr(la[0]), file=self.file)
        print("MajorVersion = " + str(la[3]), file=self.file)
        print("MinorVersion = " + str(la[4]), file=self.file)
        print("LibraryFlags = " + str(la[5]), file=self.file)
        print("LCID = " + hex(la[1]), file=self.file)
        print(file=self.file)

    def do_generate(self):
        moduleDoc = self.typelib.GetDocumentation(-1)
        stream = self.file
        docDesc = ""
        if moduleDoc[1]:
            docDesc = moduleDoc[1]
        self.progress.Starting(docDesc)
        self.progress.SetDescription("Building definitions from type library...")

        self.do_gen_file_header()

        oleItems, enumItems, recordItems, vtableItems = self.BuildOleItemsFromType()

        self.progress.SetDescription(
            "Generating...", len(oleItems) + len(enumItems) + len(vtableItems)
        )

        # Generate the constants and their support.
        if enumItems:
            print("class constants:", file=stream)
            items = list(enumItems.values())
            items.sort()
            num_written = 0
            for oleitem in items:
                num_written += oleitem.WriteEnumerationItems(stream)
                self.progress.Tick()
            if not num_written:
                print("\tpass", file=stream)
            print(file=stream)

        if self.generate_type == GEN_FULL:
            items = [l for l in oleItems.values() if l is not None]
            items.sort()
            for oleitem in items:
                self.progress.Tick()
                oleitem.WriteClass(self)

            items = list(vtableItems.values())
            items.sort()
            for oleitem in items:
                self.progress.Tick()
                oleitem.WriteClass(self)
        else:
            self.progress.Tick(len(oleItems) + len(vtableItems))

        print("RecordMap = {", file=stream)
        for record in recordItems.values():
            if record.clsid == pythoncom.IID_NULL:
                print(
                    "\t###%s: %s, # Record disabled because it doesn't have a non-null GUID"
                    % (repr(record.doc[0]), repr(str(record.clsid))),
                    file=stream,
                )
            else:
                print(
                    "\t%s: %s," % (repr(record.doc[0]), repr(str(record.clsid))),
                    file=stream,
                )
        print("}", file=stream)
        print(file=stream)

        # Write out _all_ my generated CLSID's in the map
        if self.generate_type == GEN_FULL:
            print("CLSIDToClassMap = {", file=stream)
            for item in oleItems.values():
                if item is not None and item.bWritten:
                    print(
                        "\t'%s' : %s," % (str(item.clsid), item.python_name),
                        file=stream,
                    )
            print("}", file=stream)
            print("CLSIDToPackageMap = {}", file=stream)
            print(
                "win32com.client.CLSIDToClass.RegisterCLSIDsFromDict( CLSIDToClassMap )",
                file=stream,
            )
            print("VTablesToPackageMap = {}", file=stream)
            print("VTablesToClassMap = {", file=stream)
            for item in vtableItems.values():
                print("\t'%s' : '%s'," % (item.clsid, item.python_name), file=stream)
            print("}", file=stream)
            print(file=stream)

        else:
            print("CLSIDToClassMap = {}", file=stream)
            print("CLSIDToPackageMap = {", file=stream)
            for item in oleItems.values():
                if item is not None:
                    print(
                        "\t'%s' : %s," % (str(item.clsid), repr(item.python_name)),
                        file=stream,
                    )
            print("}", file=stream)
            print("VTablesToClassMap = {}", file=stream)
            print("VTablesToPackageMap = {", file=stream)
            for item in vtableItems.values():
                print("\t'%s' : '%s'," % (item.clsid, item.python_name), file=stream)
            print("}", file=stream)
            print(file=stream)

        print(file=stream)
        # Bit of a hack - build a temp map of iteItems + vtableItems - coClasses
        map = {}
        for item in oleItems.values():
            if item is not None and not isinstance(item, CoClassItem):
                map[item.python_name] = item.clsid
        for item in vtableItems.values():  # No nones or CoClasses in this map
            map[item.python_name] = item.clsid

        print("NamesToIIDMap = {", file=stream)
        for name, iid in map.items():
            print("\t'%s' : '%s'," % (name, iid), file=stream)
        print("}", file=stream)
        print(file=stream)

        if enumItems:
            print(
                "win32com.client.constants.__dicts__.append(constants.__dict__)",
                file=stream,
            )
        print(file=stream)

    def generate_child(self, child, dir):
        "Generate a single child.  May force a few children to be built as we generate deps"
        self.generate_type = GEN_DEMAND_CHILD

        la = self.typelib.GetLibAttr()
        lcid = la[1]
        clsid = la[0]
        major = la[3]
        minor = la[4]
        self.base_mod_name = (
            "win32com.gen_py." + str(clsid)[1:-1] + "x%sx%sx%s" % (lcid, major, minor)
        )
        try:
            # Process the type library's CoClass objects, looking for the
            # specified name, or where a child has the specified name.
            # This ensures that all interesting things (including event interfaces)
            # are generated correctly.
            oleItems = {}
            vtableItems = {}
            infos = self.CollectOleItemInfosFromType()
            found = 0
            for type_info_tuple in infos:
                info, infotype, doc, attr = type_info_tuple
                if infotype == pythoncom.TKIND_COCLASS:
                    coClassItem, child_infos = self._Build_CoClass(type_info_tuple)
                    found = build.MakePublicAttributeName(doc[0]) == child
                    if not found:
                        # OK, check the child interfaces
                        for (
                            info,
                            info_type,
                            refType,
                            doc,
                            refAttr,
                            flags,
                        ) in child_infos:
                            if build.MakePublicAttributeName(doc[0]) == child:
                                found = 1
                                break
                    if found:
                        oleItems[coClassItem.clsid] = coClassItem
                        self._Build_CoClassChildren(
                            coClassItem, child_infos, oleItems, vtableItems
                        )
                        break
            if not found:
                # Doesn't appear in a class defn - look in the interface objects for it
                for type_info_tuple in infos:
                    info, infotype, doc, attr = type_info_tuple
                    if infotype in [
                        pythoncom.TKIND_INTERFACE,
                        pythoncom.TKIND_DISPATCH,
                    ]:
                        if build.MakePublicAttributeName(doc[0]) == child:
                            found = 1
                            oleItem, vtableItem = self._Build_Interface(type_info_tuple)
                            oleItems[clsid] = oleItem  # Even "None" goes in here.
                            if vtableItem is not None:
                                vtableItems[clsid] = vtableItem

            assert (
                found
            ), "Cant find the '%s' interface in the CoClasses, or the interfaces" % (
                child,
            )
            # Make a map of iid: dispitem, vtableitem)
            items = {}
            for key, value in oleItems.items():
                items[key] = (value, None)
            for key, value in vtableItems.items():
                existing = items.get(key, None)
                if existing is not None:
                    new_val = existing[0], value
                else:
                    new_val = None, value
                items[key] = new_val

            self.progress.SetDescription("Generating...", len(items))
            for oleitem, vtableitem in items.values():
                an_item = oleitem or vtableitem
                assert not self.file, "already have a file?"
                # like makepy.py, we gen to a .temp file so failure doesn't
                # leave a 1/2 generated mess.
                out_name = os.path.join(dir, an_item.python_name) + ".py"
                worked = False
                self.file = self.open_writer(out_name)
                try:
                    if oleitem is not None:
                        self.do_gen_child_item(oleitem)
                    if vtableitem is not None:
                        self.do_gen_child_item(vtableitem)
                    self.progress.Tick()
                    worked = True
                finally:
                    self.finish_writer(out_name, self.file, worked)
                    self.file = None
        finally:
            self.progress.Finished()

    def do_gen_child_item(self, oleitem):
        moduleDoc = self.typelib.GetDocumentation(-1)
        docDesc = ""
        if moduleDoc[1]:
            docDesc = moduleDoc[1]
        self.progress.Starting(docDesc)
        self.progress.SetDescription("Building definitions from type library...")
        self.do_gen_file_header()
        oleitem.WriteClass(self)
        if oleitem.bWritten:
            print(
                'win32com.client.CLSIDToClass.RegisterCLSID( "%s", %s )'
                % (oleitem.clsid, oleitem.python_name),
                file=self.file,
            )

    def checkWriteDispatchBaseClass(self):
        if not self.bHaveWrittenDispatchBaseClass:
            print("from win32com.client import DispatchBaseClass", file=self.file)
            self.bHaveWrittenDispatchBaseClass = 1

    def checkWriteCoClassBaseClass(self):
        if not self.bHaveWrittenCoClassBaseClass:
            print("from win32com.client import CoClassBaseClass", file=self.file)
            self.bHaveWrittenCoClassBaseClass = 1

    def checkWriteEventBaseClass(self):
        # Not a base class as such...
        if not self.bHaveWrittenEventBaseClass:
            # Nothing to do any more!
            self.bHaveWrittenEventBaseClass = 1


if __name__ == "__main__":
    print("This is a worker module.  Please use makepy to generate Python files.")
