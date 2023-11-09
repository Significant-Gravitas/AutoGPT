import string
import sys

import pythoncom
import win32api
from win32com.adsi import *

verbose_level = 0

server = ""  # Must have trailing /
local_name = win32api.GetComputerName()


def DumpRoot():
    "Dumps the root DSE"
    path = "LDAP://%srootDSE" % server
    rootdse = ADsGetObject(path)

    for item in rootdse.Get("SupportedLDAPVersion"):
        print("%s supports ldap version %s" % (path, item))

    attributes = ["CurrentTime", "defaultNamingContext"]
    for attr in attributes:
        val = rootdse.Get(attr)
        print(" %s=%s" % (attr, val))


###############################################
#
# Code taken from article titled:
# Reading attributeSchema and classSchema Objects
def _DumpClass(child):
    attrs = "Abstract lDAPDisplayName schemaIDGUID schemaNamingContext attributeSyntax oMSyntax"
    _DumpTheseAttributes(child, string.split(attrs))


def _DumpAttribute(child):
    attrs = "lDAPDisplayName schemaIDGUID adminDescription adminDisplayName rDNAttID defaultHidingValue defaultObjectCategory systemOnly defaultSecurityDescriptor"
    _DumpTheseAttributes(child, string.split(attrs))


def _DumpTheseAttributes(child, attrs):
    for attr in attrs:
        try:
            val = child.Get(attr)
        except pythoncom.com_error as details:
            continue
            # ###
            (hr, msg, exc, arg) = details
            if exc and exc[2]:
                msg = exc[2]
            val = "<Error: %s>" % (msg,)
        if verbose_level >= 2:
            print(" %s: %s=%s" % (child.Class, attr, val))


def DumpSchema():
    "Dumps the default DSE schema"
    # Bind to rootDSE to get the schemaNamingContext property.
    path = "LDAP://%srootDSE" % server
    rootdse = ADsGetObject(path)
    name = rootdse.Get("schemaNamingContext")

    # Bind to the actual schema container.
    path = "LDAP://" + server + name
    print("Binding to", path)
    ob = ADsGetObject(path)
    nclasses = nattr = nsub = nunk = 0

    # Enumerate the attribute and class objects in the schema container.
    for child in ob:
        # Find out if this is a class, attribute, or subSchema object.
        class_name = child.Class
        if class_name == "classSchema":
            _DumpClass(child)
            nclasses = nclasses + 1
        elif class_name == "attributeSchema":
            _DumpAttribute(child)
            nattr = nattr + 1
        elif class_name == "subSchema":
            nsub = nsub + 1
        else:
            print("Unknown class:", class_name)
            nunk = nunk + 1
    if verbose_level:
        print("Processed", nclasses, "classes")
        print("Processed", nattr, "attributes")
        print("Processed", nsub, "sub-schema's")
        print("Processed", nunk, "unknown types")


def _DumpObject(ob, level=0):
    prefix = "  " * level
    print("%s%s object: %s" % (prefix, ob.Class, ob.Name))
    # Do the directory object thing
    try:
        dir_ob = ADsGetObject(ob.ADsPath, IID_IDirectoryObject)
    except pythoncom.com_error:
        dir_ob = None
    if dir_ob is not None:
        info = dir_ob.GetObjectInformation()
        print("%s RDN='%s', ObjectDN='%s'" % (prefix, info.RDN, info.ObjectDN))
        # Create a list of names to fetch
        names = ["distinguishedName"]
        attrs = dir_ob.GetObjectAttributes(names)
        for attr in attrs:
            for val, typ in attr.Values:
                print("%s Attribute '%s' = %s" % (prefix, attr.AttrName, val))

    for child in ob:
        _DumpObject(child, level + 1)


def DumpAllObjects():
    "Recursively dump the entire directory!"
    path = "LDAP://%srootDSE" % server
    rootdse = ADsGetObject(path)
    name = rootdse.Get("defaultNamingContext")

    # Bind to the actual schema container.
    path = "LDAP://" + server + name
    print("Binding to", path)
    ob = ADsGetObject(path)

    # Enumerate the attribute and class objects in the schema container.
    _DumpObject(ob)


##########################################################
#
# Code taken from article:
# Example Code for Enumerating Schema Classes, Attributes, and Syntaxes

# Fill a map with VT_ datatypes, to give us better names:
vt_map = {}
for name, val in pythoncom.__dict__.items():
    if name[:3] == "VT_":
        vt_map[val] = name


def DumpSchema2():
    "Dumps the schema using an alternative technique"
    path = "LDAP://%sschema" % (server,)
    schema = ADsGetObject(path, IID_IADsContainer)
    nclass = nprop = nsyntax = 0
    for item in schema:
        item_class = string.lower(item.Class)
        if item_class == "class":
            items = []
            if item.Abstract:
                items.append("Abstract")
            if item.Auxiliary:
                items.append("Auxiliary")
            # 			if item.Structural: items.append("Structural")
            desc = string.join(items, ", ")
            import win32com.util

            iid_name = win32com.util.IIDToInterfaceName(item.PrimaryInterface)
            if verbose_level >= 2:
                print(
                    "Class: Name=%s, Flags=%s, Primary Interface=%s"
                    % (item.Name, desc, iid_name)
                )
            nclass = nclass + 1
        elif item_class == "property":
            if item.MultiValued:
                val_type = "Multi-Valued"
            else:
                val_type = "Single-Valued"
            if verbose_level >= 2:
                print("Property: Name=%s, %s" % (item.Name, val_type))
            nprop = nprop + 1
        elif item_class == "syntax":
            data_type = vt_map.get(item.OleAutoDataType, "<unknown type>")
            if verbose_level >= 2:
                print("Syntax: Name=%s, Datatype = %s" % (item.Name, data_type))
            nsyntax = nsyntax + 1
    if verbose_level >= 1:
        print("Processed", nclass, "classes")
        print("Processed", nprop, "properties")
        print("Processed", nsyntax, "syntax items")


def DumpGC():
    "Dumps the GC: object (whatever that is!)"
    ob = ADsGetObject("GC:", IID_IADsContainer)
    for sub_ob in ob:
        print("GC ob: %s (%s)" % (sub_ob.Name, sub_ob.ADsPath))


def DumpLocalUsers():
    "Dumps the local machine users"
    path = "WinNT://%s,computer" % (local_name,)
    ob = ADsGetObject(path, IID_IADsContainer)
    ob.put_Filter(["User", "Group"])
    for sub_ob in ob:
        print("User/Group: %s (%s)" % (sub_ob.Name, sub_ob.ADsPath))


def DumpLocalGroups():
    "Dumps the local machine groups"
    path = "WinNT://%s,computer" % (local_name,)
    ob = ADsGetObject(path, IID_IADsContainer)

    ob.put_Filter(["Group"])
    for sub_ob in ob:
        print("Group: %s (%s)" % (sub_ob.Name, sub_ob.ADsPath))
        # get the members
        members = sub_ob.Members()
        for member in members:
            print("  Group member: %s (%s)" % (member.Name, member.ADsPath))


def usage(tests):
    import os

    print("Usage: %s [-s server ] [-v] [Test ...]" % os.path.basename(sys.argv[0]))
    print("  -v : Verbose - print more information")
    print("  -s : server - execute the tests against the named server")
    print("where Test is one of:")
    for t in tests:
        print(t.__name__, ":", t.__doc__)
    print()
    print("If not tests are specified, all tests are run")
    sys.exit(1)


def main():
    import getopt
    import traceback

    tests = []
    for ob in globals().values():
        if type(ob) == type(main) and ob.__doc__:
            tests.append(ob)
    opts, args = getopt.getopt(sys.argv[1:], "s:hv")
    for opt, val in opts:
        if opt == "-s":
            if val[-1] not in "\\/":
                val = val + "/"
            global server
            server = val
        if opt == "-h":
            usage(tests)
        if opt == "-v":
            global verbose_level
            verbose_level = verbose_level + 1

    if len(args) == 0:
        print("Running all tests - use '-h' to see command-line options...")
        dotests = tests
    else:
        dotests = []
        for arg in args:
            for t in tests:
                if t.__name__ == arg:
                    dotests.append(t)
                    break
            else:
                print("Test '%s' unknown - skipping" % arg)
    if not len(dotests):
        print("Nothing to do!")
        usage(tests)
    for test in dotests:
        try:
            test()
        except:
            print("Test %s failed" % test.__name__)
            traceback.print_exc()


if __name__ == "__main__":
    main()
