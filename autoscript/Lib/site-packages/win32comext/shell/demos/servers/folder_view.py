# This is a port of the Vista SDK "FolderView" sample, and associated
# notes at http://shellrevealed.com/blogs/shellblog/archive/2007/03/15/Shell-Namespace-Extension_3A00_-Creating-and-Using-the-System-Folder-View-Object.aspx
# A key difference to shell_view.py is that this version uses the default
# IShellView provided by the shell (via SHCreateShellFolderView) rather
# than our own.
# XXX - sadly, it doesn't work quite like the original sample.  Oh well,
# another day...
import os
import pickle
import random
import sys

import commctrl
import pythoncom
import win32api
import win32con
import winerror
import winxpgui as win32gui  # the needs vista, let alone xp!
from win32com.axcontrol import axcontrol  # IObjectWithSite
from win32com.propsys import propsys
from win32com.server.exception import COMException
from win32com.server.util import NewEnum as _NewEnum, wrap as _wrap
from win32com.shell import shell, shellcon
from win32com.util import IIDToInterfaceName

GUID = pythoncom.MakeIID

# If set, output spews to the win32traceutil collector...
debug = 0


# wrap a python object in a COM pointer
def wrap(ob, iid=None):
    return _wrap(ob, iid, useDispatcher=(debug > 0))


def NewEnum(seq, iid):
    return _NewEnum(seq, iid=iid, useDispatcher=(debug > 0))


# The sample makes heavy use of "string ids" (ie, integer IDs defined in .h
# files, loaded at runtime from a (presumably localized) DLL.  We cheat.
_sids = {}  # strings, indexed bystring_id,


def LoadString(sid):
    return _sids[sid]


# fn to create a unique string ID
_last_ids = 0


def _make_ids(s):
    global _last_ids
    _last_ids += 1
    _sids[_last_ids] = s
    return _last_ids


# These strings are what the user sees and would be localized.
# XXX - its possible that the shell might persist these values, so
# this scheme wouldn't really be suitable in a real ap.
IDS_UNSPECIFIED = _make_ids("unspecified")
IDS_SMALL = _make_ids("small")
IDS_MEDIUM = _make_ids("medium")
IDS_LARGE = _make_ids("large")
IDS_CIRCLE = _make_ids("circle")
IDS_TRIANGLE = _make_ids("triangle")
IDS_RECTANGLE = _make_ids("rectangle")
IDS_POLYGON = _make_ids("polygon")
IDS_DISPLAY = _make_ids("Display")
IDS_DISPLAY_TT = _make_ids("Display the item.")
IDS_SETTINGS = _make_ids("Settings")
IDS_SETTING1 = _make_ids("Setting 1")
IDS_SETTING2 = _make_ids("Setting 2")
IDS_SETTING3 = _make_ids("Setting 3")
IDS_SETTINGS_TT = _make_ids("Modify settings.")
IDS_SETTING1_TT = _make_ids("Modify setting 1.")
IDS_SETTING2_TT = _make_ids("Modify setting 2.")
IDS_SETTING3_TT = _make_ids("Modify setting 3.")
IDS_LESSTHAN5 = _make_ids("Less Than 5")
IDS_5ORGREATER = _make_ids("Five or Greater")
del _make_ids, _last_ids

# Other misc resource stuff
IDI_ICON1 = 100
IDI_SETTINGS = 101

# The sample defines a number of "category ids".  Each one gets
# its own GUID.
CAT_GUID_NAME = GUID("{de094c9d-c65a-11dc-ba21-005056c00008}")
CAT_GUID_SIZE = GUID("{de094c9e-c65a-11dc-ba21-005056c00008}")
CAT_GUID_SIDES = GUID("{de094c9f-c65a-11dc-ba21-005056c00008}")
CAT_GUID_LEVEL = GUID("{de094ca0-c65a-11dc-ba21-005056c00008}")
# The next category guid is NOT based on a column (see
# ViewCategoryProvider::EnumCategories()...)
CAT_GUID_VALUE = "{de094ca1-c65a-11dc-ba21-005056c00008}"

GUID_Display = GUID("{4d6c2fdd-c689-11dc-ba21-005056c00008}")
GUID_Settings = GUID("{4d6c2fde-c689-11dc-ba21-005056c00008}")
GUID_Setting1 = GUID("{4d6c2fdf-c689-11dc-ba21-005056c00008}")
GUID_Setting2 = GUID("{4d6c2fe0-c689-11dc-ba21-005056c00008}")
GUID_Setting3 = GUID("{4d6c2fe1-c689-11dc-ba21-005056c00008}")

# Hrm - not sure what to do about the std keys.
# Probably need a simple parser for propkey.h
PKEY_ItemNameDisplay = ("{B725F130-47EF-101A-A5F1-02608C9EEBAC}", 10)
PKEY_PropList_PreviewDetails = ("{C9944A21-A406-48FE-8225-AEC7E24C211B}", 8)

# Not sure what the "3" here refers to - docs say PID_FIRST_USABLE (2) be
# used.  Presumably it is the 'propID' value in the .propdesc file!
# note that the following GUIDs are also references in the .propdesc file
PID_SOMETHING = 3
# These are our 'private' PKEYs
# Col 2, name="Sample.AreaSize"
PKEY_Sample_AreaSize = ("{d6f5e341-c65c-11dc-ba21-005056c00008}", PID_SOMETHING)
# Col 3, name="Sample.NumberOfSides"
PKEY_Sample_NumberOfSides = ("{d6f5e342-c65c-11dc-ba21-005056c00008}", PID_SOMETHING)
# Col 4, name="Sample.DirectoryLevel"
PKEY_Sample_DirectoryLevel = ("{d6f5e343-c65c-11dc-ba21-005056c00008}", PID_SOMETHING)


# We construct a PIDL from a pickle of a dict - turn it back into a
# dict (we should *never* be called with a PIDL that the last elt is not
# ours, so it is safe to assume we created it (assume->"ass" = "u" + "me" :)
def pidl_to_item(pidl):
    # Note that only the *last* elt in the PIDL is certainly ours,
    # but it contains everything we need encoded as a dict.
    return pickle.loads(pidl[-1])


# Start of msdn sample port...
# make_item_enum replaces the sample's entire EnumIDList.cpp :)
def make_item_enum(level, flags):
    pidls = []
    nums = """zero one two three four five size seven eight nine ten""".split()
    for i, name in enumerate(nums):
        size = random.randint(0, 255)
        sides = 1
        while sides in [1, 2]:
            sides = random.randint(0, 5)
        is_folder = (i % 2) != 0
        # check the flags say to include it.
        # (This seems strange; if you ask the same folder for, but appear
        skip = False
        if not (flags & shellcon.SHCONTF_STORAGE):
            if is_folder:
                skip = not (flags & shellcon.SHCONTF_FOLDERS)
            else:
                skip = not (flags & shellcon.SHCONTF_NONFOLDERS)
        if not skip:
            data = dict(
                name=name, size=size, sides=sides, level=level, is_folder=is_folder
            )
            pidls.append([pickle.dumps(data)])
    return NewEnum(pidls, shell.IID_IEnumIDList)


# start of Utils.cpp port
def DisplayItem(shell_item_array, hwnd_parent=0):
    # Get the first ShellItem and display its name
    if shell_item_array is None:
        msg = "You must select something!"
    else:
        si = shell_item_array.GetItemAt(0)
        name = si.GetDisplayName(shellcon.SIGDN_NORMALDISPLAY)
        msg = "%d items selected, first is %r" % (shell_item_array.GetCount(), name)
    win32gui.MessageBox(hwnd_parent, msg, "Hello", win32con.MB_OK)


# end of Utils.cpp port


# start of sample's FVCommands.cpp port
class Command:
    def __init__(self, guid, ids, ids_tt, idi, flags, callback, children):
        self.guid = guid
        self.ids = ids
        self.ids_tt = ids_tt
        self.idi = idi
        self.flags = flags
        self.callback = callback
        self.children = children
        assert not children or isinstance(children[0], Command)

    def tuple(self):
        return (
            self.guid,
            self.ids,
            self.ids_tt,
            self.idi,
            self.flags,
            self.callback,
            self.children,
        )


# command callbacks - called back directly by us - see ExplorerCommand.Invoke
def onDisplay(items, bindctx):
    DisplayItem(items)


def onSetting1(items, bindctx):
    win32gui.MessageBox(0, LoadString(IDS_SETTING1), "Hello", win32con.MB_OK)


def onSetting2(items, bindctx):
    win32gui.MessageBox(0, LoadString(IDS_SETTING2), "Hello", win32con.MB_OK)


def onSetting3(items, bindctx):
    win32gui.MessageBox(0, LoadString(IDS_SETTING3), "Hello", win32con.MB_OK)


taskSettings = [
    Command(
        GUID_Setting1, IDS_SETTING1, IDS_SETTING1_TT, IDI_SETTINGS, 0, onSetting1, None
    ),
    Command(
        GUID_Setting2, IDS_SETTING2, IDS_SETTING2_TT, IDI_SETTINGS, 0, onSetting2, None
    ),
    Command(
        GUID_Setting3, IDS_SETTING3, IDS_SETTING3_TT, IDI_SETTINGS, 0, onSetting3, None
    ),
]

tasks = [
    Command(GUID_Display, IDS_DISPLAY, IDS_DISPLAY_TT, IDI_ICON1, 0, onDisplay, None),
    Command(
        GUID_Settings,
        IDS_SETTINGS,
        IDS_SETTINGS_TT,
        IDI_SETTINGS,
        shellcon.ECF_HASSUBCOMMANDS,
        None,
        taskSettings,
    ),
]


class ExplorerCommandProvider:
    _com_interfaces_ = [shell.IID_IExplorerCommandProvider]
    _public_methods_ = shellcon.IExplorerCommandProvider_Methods

    def GetCommands(self, site, iid):
        items = [wrap(ExplorerCommand(t)) for t in tasks]
        return NewEnum(items, shell.IID_IEnumExplorerCommand)


class ExplorerCommand:
    _com_interfaces_ = [shell.IID_IExplorerCommand]
    _public_methods_ = shellcon.IExplorerCommand_Methods

    def __init__(self, cmd):
        self.cmd = cmd

    # The sample also appears to ignore the pidl args!?
    def GetTitle(self, pidl):
        return LoadString(self.cmd.ids)

    def GetToolTip(self, pidl):
        return LoadString(self.cmd.ids_tt)

    def GetIcon(self, pidl):
        # Return a string of the usual "dll,resource_id" format
        # todo - just return any ".ico that comes with python" + ",0" :)
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetState(self, shell_items, slow_ok):
        return shellcon.ECS_ENABLED

    def GetFlags(self):
        return self.cmd.flags

    def GetCanonicalName(self):
        return self.cmd.guid

    def Invoke(self, items, bind_ctx):
        # If no function defined - just return S_OK
        if self.cmd.callback:
            self.cmd.callback(items, bind_ctx)
        else:
            print("No callback for command ", LoadString(self.cmd.ids))

    def EnumSubCommands(self):
        if not self.cmd.children:
            return None
        items = [wrap(ExplorerCommand(c)) for c in self.cmd.children]
        return NewEnum(items, shell.IID_IEnumExplorerCommand)


# end of sample's FVCommands.cpp port


# start of sample's Category.cpp port
class FolderViewCategorizer:
    _com_interfaces_ = [shell.IID_ICategorizer]
    _public_methods_ = shellcon.ICategorizer_Methods

    description = None  # subclasses should set their own

    def __init__(self, shell_folder):
        self.sf = shell_folder

    #  Determines the relative order of two items in their item identifier lists.
    def CompareCategory(self, flags, cat1, cat2):
        return cat1 - cat2

    #  Retrieves the name of a categorizer, such as "Group By Device
    #  Type", that can be displayed in the user interface.
    def GetDescription(self, cch):
        return self.description

    # Retrieves information about a category, such as the default
    # display and the text to display in the user interface.
    def GetCategoryInfo(self, catid):
        # Note: this isn't always appropriate!  See overrides below
        return 0, str(catid)  # ????


class FolderViewCategorizer_Name(FolderViewCategorizer):
    description = "Alphabetical"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            ret.append(val)
        return ret


class FolderViewCategorizer_Size(FolderViewCategorizer):
    description = "Group By Size"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            # Why don't we just get the size of the PIDL?
            val = self.sf.GetDetailsEx(pidl, PKEY_Sample_AreaSize)
            val = int(val)  # it probably came in a VT_BSTR variant
            if val < 255 // 3:
                cid = IDS_SMALL
            elif val < 2 * 255 // 3:
                cid = IDS_MEDIUM
            else:
                cid = IDS_LARGE
            ret.append(cid)
        return ret

    def GetCategoryInfo(self, catid):
        return 0, LoadString(catid)


class FolderViewCategorizer_Sides(FolderViewCategorizer):
    description = "Group By Sides"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            if val == 0:
                cid = IDS_CIRCLE
            elif val == 3:
                cid = IDS_TRIANGLE
            elif val == 4:
                cid = IDS_RECTANGLE
            elif val == 5:
                cid = IDS_POLYGON
            else:
                cid = IDS_UNSPECIFIED
            ret.append(cid)
        return ret

    def GetCategoryInfo(self, catid):
        return 0, LoadString(catid)


class FolderViewCategorizer_Value(FolderViewCategorizer):
    description = "Group By Value"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            if val in "one two three four".split():
                ret.append(IDS_LESSTHAN5)
            else:
                ret.append(IDS_5ORGREATER)
        return ret

    def GetCategoryInfo(self, catid):
        return 0, LoadString(catid)


class FolderViewCategorizer_Level(FolderViewCategorizer):
    description = "Group By Value"

    def GetCategory(self, pidls):
        return [
            self.sf.GetDetailsEx(pidl, PKEY_Sample_DirectoryLevel) for pidl in pidls
        ]


class ViewCategoryProvider:
    _com_interfaces_ = [shell.IID_ICategoryProvider]
    _public_methods_ = shellcon.ICategoryProvider_Methods

    def __init__(self, shell_folder):
        self.shell_folder = shell_folder

    def CanCategorizeOnSCID(self, pkey):
        return pkey in [
            PKEY_ItemNameDisplay,
            PKEY_Sample_AreaSize,
            PKEY_Sample_NumberOfSides,
            PKEY_Sample_DirectoryLevel,
        ]

    #  Creates a category object.
    def CreateCategory(self, guid, iid):
        if iid == shell.IID_ICategorizer:
            if guid == CAT_GUID_NAME:
                klass = FolderViewCategorizer_Name
            elif guid == CAT_GUID_SIDES:
                klass = FolderViewCategorizer_Sides
            elif guid == CAT_GUID_SIZE:
                klass = FolderViewCategorizer_Size
            elif guid == CAT_GUID_VALUE:
                klass = FolderViewCategorizer_Value
            elif guid == CAT_GUID_LEVEL:
                klass = FolderViewCategorizer_Level
            else:
                raise COMException(hresult=winerror.E_INVALIDARG)
            return wrap(klass(self.shell_folder))
        raise COMException(hresult=winerror.E_NOINTERFACE)

    #  Retrieves the enumerator for the categories.
    def EnumCategories(self):
        # These are additional categories beyond the columns
        seq = [CAT_GUID_VALUE]
        return NewEnum(seq, pythoncom.IID_IEnumGUID)

    #  Retrieves a globally unique identifier (GUID) that represents
    #  the categorizer to use for the specified Shell column.
    def GetCategoryForSCID(self, scid):
        if scid == PKEY_ItemNameDisplay:
            guid = CAT_GUID_NAME
        elif scid == PKEY_Sample_AreaSize:
            guid = CAT_GUID_SIZE
        elif scid == PKEY_Sample_NumberOfSides:
            guid = CAT_GUID_SIDES
        elif scid == PKEY_Sample_DirectoryLevel:
            guid = CAT_GUID_LEVEL
        elif scid == pythoncom.IID_NULL:
            # This can be called with a NULL
            # format ID. This will happen if you have a category,
            # not based on a column, that gets stored in the
            # property bag. When a return is made to this item,
            # it will call this function with a NULL format id.
            guid = CAT_GUID_VALUE
        else:
            raise COMException(hresult=winerror.E_INVALIDARG)
        return guid

    #  Retrieves the name of the specified category. This is where
    #  additional categories that appear under the column
    #  related categories in the UI, get their display names.
    def GetCategoryName(self, guid, cch):
        if guid == CAT_GUID_VALUE:
            return "Value"
        raise COMException(hresult=winerror.E_FAIL)

    #  Enables the folder to override the default grouping.
    def GetDefaultCategory(self):
        return CAT_GUID_LEVEL, (pythoncom.IID_NULL, 0)


# end of sample's Category.cpp port

# start of sample's ContextMenu.cpp port
MENUVERB_DISPLAY = 0

folderViewImplContextMenuIDs = [
    (
        "display",
        MENUVERB_DISPLAY,
        0,
    ),
]


class ContextMenu:
    _reg_progid_ = "Python.ShellFolderSample.ContextMenu"
    _reg_desc_ = "Python FolderView Context Menu"
    _reg_clsid_ = "{fed40039-021f-4011-87c5-6188b9979764}"
    _com_interfaces_ = [
        shell.IID_IShellExtInit,
        shell.IID_IContextMenu,
        axcontrol.IID_IObjectWithSite,
    ]
    _public_methods_ = (
        shellcon.IContextMenu_Methods
        + shellcon.IShellExtInit_Methods
        + ["GetSite", "SetSite"]
    )
    _context_menu_type_ = "PythonFolderViewSampleType"

    def __init__(self):
        self.site = None
        self.dataobj = None

    def Initialize(self, folder, dataobj, hkey):
        self.dataobj = dataobj

    def QueryContextMenu(self, hMenu, indexMenu, idCmdFirst, idCmdLast, uFlags):
        s = LoadString(IDS_DISPLAY)
        win32gui.InsertMenu(
            hMenu, indexMenu, win32con.MF_BYPOSITION, idCmdFirst + MENUVERB_DISPLAY, s
        )
        indexMenu += 1
        # other verbs could go here...

        # indicate that we added one verb.
        return 1

    def InvokeCommand(self, ci):
        mask, hwnd, verb, params, dir, nShow, hotkey, hicon = ci
        # this seems very convuluted, but its what the sample does :)
        for verb_name, verb_id, flag in folderViewImplContextMenuIDs:
            if isinstance(verb, int):
                matches = verb == verb_id
            else:
                matches = verb == verb_name
            if matches:
                break
        else:
            assert False, ci  # failed to find our ID
        if verb_id == MENUVERB_DISPLAY:
            sia = shell.SHCreateShellItemArrayFromDataObject(self.dataobj)
            DisplayItem(hwnd, sia)
        else:
            assert False, ci  # Got some verb we weren't expecting?

    def GetCommandString(self, cmd, typ):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetSite(self, site):
        self.site = site

    def GetSite(self, iid):
        return self.site


# end of sample's ContextMenu.cpp port


# start of sample's ShellFolder.cpp port
class ShellFolder:
    _com_interfaces_ = [
        shell.IID_IBrowserFrameOptions,
        pythoncom.IID_IPersist,
        shell.IID_IPersistFolder,
        shell.IID_IPersistFolder2,
        shell.IID_IShellFolder,
        shell.IID_IShellFolder2,
    ]

    _public_methods_ = (
        shellcon.IBrowserFrame_Methods
        + shellcon.IPersistFolder2_Methods
        + shellcon.IShellFolder2_Methods
    )

    _reg_progid_ = "Python.ShellFolderSample.Folder2"
    _reg_desc_ = "Python FolderView sample"
    _reg_clsid_ = "{bb8c24ad-6aaa-4cec-ac5e-c429d5f57627}"

    max_levels = 5

    def __init__(self, level=0):
        self.current_level = level
        self.pidl = None  # set when Initialize is called

    def ParseDisplayName(self, hwnd, reserved, displayName, attr):
        # print "ParseDisplayName", displayName
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumObjects(self, hwndOwner, flags):
        if self.current_level >= self.max_levels:
            return None
        return make_item_enum(self.current_level + 1, flags)

    def BindToObject(self, pidl, bc, iid):
        tail = pidl_to_item(pidl)
        # assert tail['is_folder'], "BindToObject should only be called on folders?"
        # *sob*
        # No point creating object just to have QI fail.
        if iid not in ShellFolder._com_interfaces_:
            raise COMException(hresult=winerror.E_NOTIMPL)
        child = ShellFolder(self.current_level + 1)
        # hrmph - not sure what multiple PIDLs here mean?
        #        assert len(pidl)==1, pidl # expecting just relative child PIDL
        child.Initialize(self.pidl + pidl)
        return wrap(child, iid)

    def BindToStorage(self, pidl, bc, iid):
        return self.BindToObject(pidl, bc, iid)

    def CompareIDs(self, param, id1, id2):
        return 0  # XXX - todo - implement this!

    def CreateViewObject(self, hwnd, iid):
        if iid == shell.IID_IShellView:
            com_folder = wrap(self)
            return shell.SHCreateShellFolderView(com_folder)
        elif iid == shell.IID_ICategoryProvider:
            return wrap(ViewCategoryProvider(self))
        elif iid == shell.IID_IContextMenu:
            ws = wrap(self)
            dcm = (hwnd, None, self.pidl, ws, None)
            return shell.SHCreateDefaultContextMenu(dcm, iid)
        elif iid == shell.IID_IExplorerCommandProvider:
            return wrap(ExplorerCommandProvider())
        else:
            raise COMException(hresult=winerror.E_NOINTERFACE)

    def GetAttributesOf(self, pidls, attrFlags):
        assert len(pidls) == 1, "sample only expects 1 too!"
        assert len(pidls[0]) == 1, "expect relative pidls!"
        item = pidl_to_item(pidls[0])
        flags = 0
        if item["is_folder"]:
            flags |= shellcon.SFGAO_FOLDER
        if item["level"] < self.max_levels:
            flags |= shellcon.SFGAO_HASSUBFOLDER
        return flags

    #  Retrieves an OLE interface that can be used to carry out
    #  actions on the specified file objects or folders.
    def GetUIObjectOf(self, hwndOwner, pidls, iid, inout):
        assert len(pidls) == 1, "oops - arent expecting more than one!"
        assert len(pidls[0]) == 1, "assuming relative pidls!"
        item = pidl_to_item(pidls[0])
        if iid == shell.IID_IContextMenu:
            ws = wrap(self)
            dcm = (hwndOwner, None, self.pidl, ws, pidls)
            return shell.SHCreateDefaultContextMenu(dcm, iid)
        elif iid == shell.IID_IExtractIconW:
            dxi = shell.SHCreateDefaultExtractIcon()
            # dxi is IDefaultExtractIconInit
            if item["is_folder"]:
                dxi.SetNormalIcon("shell32.dll", 4)
            else:
                dxi.SetNormalIcon("shell32.dll", 1)
            # just return the dxi - let Python QI for IID_IExtractIconW
            return dxi

        elif iid == pythoncom.IID_IDataObject:
            return shell.SHCreateDataObject(self.pidl, pidls, None, iid)

        elif iid == shell.IID_IQueryAssociations:
            elts = []
            if item["is_folder"]:
                elts.append((shellcon.ASSOCCLASS_FOLDER, None, None))
            elts.append(
                (shellcon.ASSOCCLASS_PROGID_STR, None, ContextMenu._context_menu_type_)
            )
            return shell.AssocCreateForClasses(elts, iid)

        raise COMException(hresult=winerror.E_NOINTERFACE)

    #  Retrieves the display name for the specified file object or subfolder.
    def GetDisplayNameOf(self, pidl, flags):
        item = pidl_to_item(pidl)
        if flags & shellcon.SHGDN_FORPARSING:
            if flags & shellcon.SHGDN_INFOLDER:
                return item["name"]
            else:
                if flags & shellcon.SHGDN_FORADDRESSBAR:
                    sigdn = shellcon.SIGDN_DESKTOPABSOLUTEEDITING
                else:
                    sigdn = shellcon.SIGDN_DESKTOPABSOLUTEPARSING
                parent = shell.SHGetNameFromIDList(self.pidl, sigdn)
                return parent + "\\" + item["name"]
        else:
            return item["name"]

    def SetNameOf(self, hwndOwner, pidl, new_name, flags):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetClassID(self):
        return self._reg_clsid_

    #  IPersistFolder method
    def Initialize(self, pidl):
        self.pidl = pidl

    #  IShellFolder2 methods
    def EnumSearches(self):
        raise COMException(hresult=winerror.E_NOINTERFACE)

    #  Retrieves the default sorting and display columns.
    def GetDefaultColumn(self, dwres):
        # result is (sort, display)
        return 0, 0

    #  Retrieves the default state for a specified column.
    def GetDefaultColumnState(self, iCol):
        if iCol < 3:
            return shellcon.SHCOLSTATE_ONBYDEFAULT | shellcon.SHCOLSTATE_TYPE_STR
        raise COMException(hresult=winerror.E_INVALIDARG)

    #  Requests the GUID of the default search object for the folder.
    def GetDefaultSearchGUID(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    #  Helper function for getting the display name for a column.
    def _GetColumnDisplayName(self, pidl, pkey):
        item = pidl_to_item(pidl)
        is_folder = item["is_folder"]
        if pkey == PKEY_ItemNameDisplay:
            val = item["name"]
        elif pkey == PKEY_Sample_AreaSize and not is_folder:
            val = "%d Sq. Ft." % item["size"]
        elif pkey == PKEY_Sample_NumberOfSides and not is_folder:
            val = str(item["sides"])  # not sure why str()
        elif pkey == PKEY_Sample_DirectoryLevel:
            val = str(item["level"])
        else:
            val = ""
        return val

    #  Retrieves detailed information, identified by a
    #  property set ID (FMTID) and property ID (PID),
    #  on an item in a Shell folder.
    def GetDetailsEx(self, pidl, pkey):
        item = pidl_to_item(pidl)
        is_folder = item["is_folder"]
        if not is_folder and pkey == PKEY_PropList_PreviewDetails:
            return "prop:Sample.AreaSize;Sample.NumberOfSides;Sample.DirectoryLevel"
        return self._GetColumnDisplayName(pidl, pkey)

    #  Retrieves detailed information, identified by a
    #  column index, on an item in a Shell folder.
    def GetDetailsOf(self, pidl, iCol):
        key = self.MapColumnToSCID(iCol)
        if pidl is None:
            data = [
                (commctrl.LVCFMT_LEFT, "Name"),
                (commctrl.LVCFMT_CENTER, "Size"),
                (commctrl.LVCFMT_CENTER, "Sides"),
                (commctrl.LVCFMT_CENTER, "Level"),
            ]
            if iCol >= len(data):
                raise COMException(hresult=winerror.E_FAIL)
            fmt, val = data[iCol]
        else:
            fmt = 0  # ?
            val = self._GetColumnDisplayName(pidl, key)
        cxChar = 24
        return fmt, cxChar, val

    #  Converts a column name to the appropriate
    #  property set ID (FMTID) and property ID (PID).
    def MapColumnToSCID(self, iCol):
        data = [
            PKEY_ItemNameDisplay,
            PKEY_Sample_AreaSize,
            PKEY_Sample_NumberOfSides,
            PKEY_Sample_DirectoryLevel,
        ]
        if iCol >= len(data):
            raise COMException(hresult=winerror.E_FAIL)
        return data[iCol]

    #  IPersistFolder2 methods
    #  Retrieves the PIDLIST_ABSOLUTE for the folder object.
    def GetCurFolder(self):
        # The docs say this is OK, but I suspect its a problem in this case :)
        # assert self.pidl, "haven't been initialized?"
        return self.pidl


# end of sample's ShellFolder.cpp port


def get_schema_fname():
    me = win32api.GetFullPathName(__file__)
    sc = os.path.splitext(me)[0] + ".propdesc"
    assert os.path.isfile(sc), sc
    return sc


def DllRegisterServer():
    import winreg

    if sys.getwindowsversion()[0] < 6:
        print("This sample only works on Vista")
        sys.exit(1)

    key = winreg.CreateKey(
        winreg.HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\"
        "Explorer\\Desktop\\Namespace\\" + ShellFolder._reg_clsid_,
    )
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellFolder._reg_desc_)
    # And special shell keys under our CLSID
    key = winreg.CreateKey(
        winreg.HKEY_CLASSES_ROOT, "CLSID\\" + ShellFolder._reg_clsid_ + "\\ShellFolder"
    )
    # 'Attributes' is an int stored as a binary! use struct
    attr = (
        shellcon.SFGAO_FOLDER | shellcon.SFGAO_HASSUBFOLDER | shellcon.SFGAO_BROWSABLE
    )
    import struct

    s = struct.pack("i", attr)
    winreg.SetValueEx(key, "Attributes", 0, winreg.REG_BINARY, s)
    # register the context menu handler under the FolderViewSampleType type.
    keypath = "%s\\shellex\\ContextMenuHandlers\\%s" % (
        ContextMenu._context_menu_type_,
        ContextMenu._reg_desc_,
    )
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, keypath)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ContextMenu._reg_clsid_)
    propsys.PSRegisterPropertySchema(get_schema_fname())
    print(ShellFolder._reg_desc_, "registration complete.")


def DllUnregisterServer():
    import winreg

    paths = [
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Desktop\\Namespace\\"
        + ShellFolder._reg_clsid_,
        "%s\\shellex\\ContextMenuHandlers\\%s"
        % (ContextMenu._context_menu_type_, ContextMenu._reg_desc_),
    ]
    for path in paths:
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, path)
        except WindowsError as details:
            import errno

            if details.errno != errno.ENOENT:
                print("FAILED to remove %s: %s" % (path, details))

    propsys.PSUnregisterPropertySchema(get_schema_fname())
    print(ShellFolder._reg_desc_, "unregistration complete.")


if __name__ == "__main__":
    from win32com.server import register

    register.UseCommandLine(
        ShellFolder,
        ContextMenu,
        debug=debug,
        finalize_register=DllRegisterServer,
        finalize_unregister=DllUnregisterServer,
    )
