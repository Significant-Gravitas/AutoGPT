import pythoncom
import pywintypes
import win32security
from win32com.adsi import adsi, adsicon
from win32com.adsi.adsicon import *

options = None  # set to optparse options object

ADsTypeNameMap = {}


def getADsTypeName(type_val):
    # convert integer type to the 'typename' as known in the headerfiles.
    if not ADsTypeNameMap:
        for n, v in adsicon.__dict__.items():
            if n.startswith("ADSTYPE_"):
                ADsTypeNameMap[v] = n
    return ADsTypeNameMap.get(type_val, hex(type_val))


def _guid_from_buffer(b):
    return pywintypes.IID(b, True)


def _sid_from_buffer(b):
    return str(pywintypes.SID(b))


_null_converter = lambda x: x

converters = {
    "objectGUID": _guid_from_buffer,
    "objectSid": _sid_from_buffer,
    "instanceType": getADsTypeName,
}


def log(level, msg, *args):
    if options.verbose >= level:
        print("log:", msg % args)


def getGC():
    cont = adsi.ADsOpenObject(
        "GC:", options.user, options.password, 0, adsi.IID_IADsContainer
    )
    enum = adsi.ADsBuildEnumerator(cont)
    # Only 1 child of the global catalog.
    for e in enum:
        gc = e.QueryInterface(adsi.IID_IDirectorySearch)
        return gc
    return None


def print_attribute(col_data):
    prop_name, prop_type, values = col_data
    if values is not None:
        log(2, "property '%s' has type '%s'", prop_name, getADsTypeName(prop_type))
        value = [converters.get(prop_name, _null_converter)(v[0]) for v in values]
        if len(value) == 1:
            value = value[0]
        print(" %s=%r" % (prop_name, value))
    else:
        print(" %s is None" % (prop_name,))


def search():
    gc = getGC()
    if gc is None:
        log(0, "Can't find the global catalog")
        return

    prefs = [(ADS_SEARCHPREF_SEARCH_SCOPE, (ADS_SCOPE_SUBTREE,))]
    hr, statuses = gc.SetSearchPreference(prefs)
    log(3, "SetSearchPreference returned %d/%r", hr, statuses)

    if options.attributes:
        attributes = options.attributes.split(",")
    else:
        attributes = None

    h = gc.ExecuteSearch(options.filter, attributes)
    hr = gc.GetNextRow(h)
    while hr != S_ADS_NOMORE_ROWS:
        print("-- new row --")
        if attributes is None:
            # Loop over all columns returned
            while 1:
                col_name = gc.GetNextColumnName(h)
                if col_name is None:
                    break
                data = gc.GetColumn(h, col_name)
                print_attribute(data)
        else:
            # loop over attributes specified.
            for a in attributes:
                try:
                    data = gc.GetColumn(h, a)
                    print_attribute(data)
                except adsi.error as details:
                    if details[0] != E_ADS_COLUMN_NOT_SET:
                        raise
                    print_attribute((a, None, None))
        hr = gc.GetNextRow(h)
    gc.CloseSearchHandle(h)


def main():
    global options
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option(
        "-f", "--file", dest="filename", help="write report to FILE", metavar="FILE"
    )
    parser.add_option(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase verbosity of output",
    )
    parser.add_option(
        "-q", "--quiet", action="store_true", help="suppress output messages"
    )

    parser.add_option("-U", "--user", help="specify the username used to connect")
    parser.add_option("-P", "--password", help="specify the password used to connect")
    parser.add_option(
        "",
        "--filter",
        default="(&(objectCategory=person)(objectClass=User))",
        help="specify the search filter",
    )
    parser.add_option(
        "", "--attributes", help="comma sep'd list of attribute names to print"
    )

    options, args = parser.parse_args()
    if options.quiet:
        if options.verbose != 1:
            parser.error("Can not use '--verbose' and '--quiet'")
        options.verbose = 0

    if args:
        parser.error("You need not specify args")

    search()


if __name__ == "__main__":
    main()
