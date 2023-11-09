"""Utilities for selecting and enumerating the Type Libraries installed on the system
"""

import pythoncom
import win32api
import win32con


class TypelibSpec:
    def __init__(self, clsid, lcid, major, minor, flags=0):
        self.clsid = str(clsid)
        self.lcid = int(lcid)
        # We avoid assuming 'major' or 'minor' are integers - when
        # read from the registry there is some confusion about if
        # they are base 10 or base 16 (they *should* be base 16, but
        # how they are written is beyond our control.)
        self.major = major
        self.minor = minor
        self.dll = None
        self.desc = None
        self.ver_desc = None
        self.flags = flags

    # For the SelectList
    def __getitem__(self, item):
        if item == 0:
            return self.ver_desc
        raise IndexError("Cant index me!")

    def __lt__(self, other):  # rich-cmp/py3k-friendly version
        me = (
            (self.ver_desc or "").lower(),
            (self.desc or "").lower(),
            self.major,
            self.minor,
        )
        them = (
            (other.ver_desc or "").lower(),
            (other.desc or "").lower(),
            other.major,
            other.minor,
        )
        return me < them

    def __eq__(self, other):  # rich-cmp/py3k-friendly version
        return (
            (self.ver_desc or "").lower() == (other.ver_desc or "").lower()
            and (self.desc or "").lower() == (other.desc or "").lower()
            and self.major == other.major
            and self.minor == other.minor
        )

    def Resolve(self):
        if self.dll is None:
            return 0
        tlb = pythoncom.LoadTypeLib(self.dll)
        self.FromTypelib(tlb, None)
        return 1

    def FromTypelib(self, typelib, dllName=None):
        la = typelib.GetLibAttr()
        self.clsid = str(la[0])
        self.lcid = la[1]
        self.major = la[3]
        self.minor = la[4]
        if dllName:
            self.dll = dllName


def EnumKeys(root):
    index = 0
    ret = []
    while 1:
        try:
            item = win32api.RegEnumKey(root, index)
        except win32api.error:
            break
        try:
            # Note this doesn't handle REG_EXPAND_SZ, but the implementation
            # here doesn't need to - that is handled as the data is read.
            val = win32api.RegQueryValue(root, item)
        except win32api.error:
            val = ""  # code using this assumes a string.

        ret.append((item, val))
        index = index + 1
    return ret


FLAG_RESTRICTED = 1
FLAG_CONTROL = 2
FLAG_HIDDEN = 4


def EnumTlbs(excludeFlags=0):
    """Return a list of TypelibSpec objects, one for each registered library."""
    key = win32api.RegOpenKey(win32con.HKEY_CLASSES_ROOT, "Typelib")
    iids = EnumKeys(key)
    results = []
    for iid, crap in iids:
        try:
            key2 = win32api.RegOpenKey(key, str(iid))
        except win32api.error:
            # A few good reasons for this, including "access denied".
            continue
        for version, tlbdesc in EnumKeys(key2):
            major_minor = version.split(".", 1)
            if len(major_minor) < 2:
                major_minor.append("0")
            # For some reason, this code used to assume the values were hex.
            # This seems to not be true - particularly for CDO 1.21
            # *sigh* - it appears there are no rules here at all, so when we need
            # to know the info, we must load the tlb by filename and request it.
            # The Resolve() method on the TypelibSpec does this.
            # For this reason, keep the version numbers as strings - that
            # way we can't be wrong!  Let code that really needs an int to work
            # out what to do.  FWIW, http://support.microsoft.com/kb/816970 is
            # pretty clear that they *should* be hex.
            major = major_minor[0]
            minor = major_minor[1]
            key3 = win32api.RegOpenKey(key2, str(version))
            try:
                # The "FLAGS" are at this point
                flags = int(win32api.RegQueryValue(key3, "FLAGS"))
            except (win32api.error, ValueError):
                flags = 0
            if flags & excludeFlags == 0:
                for lcid, crap in EnumKeys(key3):
                    try:
                        lcid = int(lcid)
                    except ValueError:  # not an LCID entry
                        continue
                    # Check for both "{lcid}\win32" and "{lcid}\win64" keys.
                    try:
                        key4 = win32api.RegOpenKey(key3, "%s\\win32" % (lcid,))
                    except win32api.error:
                        try:
                            key4 = win32api.RegOpenKey(key3, "%s\\win64" % (lcid,))
                        except win32api.error:
                            continue
                    try:
                        dll, typ = win32api.RegQueryValueEx(key4, None)
                        if typ == win32con.REG_EXPAND_SZ:
                            dll = win32api.ExpandEnvironmentStrings(dll)
                    except win32api.error:
                        dll = None
                    spec = TypelibSpec(iid, lcid, major, minor, flags)
                    spec.dll = dll
                    spec.desc = tlbdesc
                    spec.ver_desc = tlbdesc + " (" + version + ")"
                    results.append(spec)
    return results


def FindTlbsWithDescription(desc):
    """Find all installed type libraries with the specified description"""
    ret = []
    items = EnumTlbs()
    for item in items:
        if item.desc == desc:
            ret.append(item)
    return ret


def SelectTlb(title="Select Library", excludeFlags=0):
    """Display a list of all the type libraries, and select one.   Returns None if cancelled"""
    import pywin.dialogs.list

    items = EnumTlbs(excludeFlags)
    # fixup versions - we assume hex (see __init__ above)
    for i in items:
        i.major = int(i.major, 16)
        i.minor = int(i.minor, 16)
    items.sort()
    rc = pywin.dialogs.list.SelectFromLists(title, items, ["Type Library"])
    if rc is None:
        return None
    return items[rc]


# Test code.
if __name__ == "__main__":
    print(SelectTlb().__dict__)
