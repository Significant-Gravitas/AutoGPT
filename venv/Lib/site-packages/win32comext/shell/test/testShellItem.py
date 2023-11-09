# Test IShellItem and related interfaces
import unittest

from win32com.shell import knownfolders, shell, shellcon


class TestShellItem(unittest.TestCase):
    def assertShellItemsEqual(self, i1, i2):
        n1 = i1.GetDisplayName(shellcon.SHGDN_FORPARSING)
        n2 = i2.GetDisplayName(shellcon.SHGDN_FORPARSING)
        self.assertEqual(n1, n2)

    def test_idlist_roundtrip(self):
        pidl = shell.SHGetSpecialFolderLocation(0, shellcon.CSIDL_DESKTOP)
        item = shell.SHCreateItemFromIDList(pidl, shell.IID_IShellItem)
        pidl_back = shell.SHGetIDListFromObject(item)
        self.assertEqual(pidl, pidl_back)

    def test_parsing_name(self):
        sf = shell.SHGetDesktopFolder()
        flags = shellcon.SHCONTF_FOLDERS | shellcon.SHCONTF_NONFOLDERS
        children = sf.EnumObjects(0, flags)
        child_pidl = next(children)
        name = sf.GetDisplayNameOf(child_pidl, shellcon.SHGDN_FORPARSING)

        item = shell.SHCreateItemFromParsingName(name, None, shell.IID_IShellItem)
        # test the name we get from the item is the same as from the folder.
        self.assertEqual(name, item.GetDisplayName(shellcon.SHGDN_FORPARSING))

    def test_parsing_relative(self):
        desktop_pidl = shell.SHGetSpecialFolderLocation(0, shellcon.CSIDL_DESKTOP)
        desktop_item = shell.SHCreateItemFromIDList(desktop_pidl, shell.IID_IShellItem)

        sf = shell.SHGetDesktopFolder()
        flags = shellcon.SHCONTF_FOLDERS | shellcon.SHCONTF_NONFOLDERS
        children = sf.EnumObjects(0, flags)
        child_pidl = next(children)
        name_flags = shellcon.SHGDN_FORPARSING | shellcon.SHGDN_INFOLDER
        name = sf.GetDisplayNameOf(child_pidl, name_flags)

        item = shell.SHCreateItemFromRelativeName(
            desktop_item, name, None, shell.IID_IShellItem
        )
        # test the name we get from the item is the same as from the folder.
        self.assertEqual(name, item.GetDisplayName(name_flags))

    def test_create_in_known_folder(self):
        item = shell.SHCreateItemInKnownFolder(
            knownfolders.FOLDERID_Desktop, 0, None, shell.IID_IShellItem
        )
        # this will do for now :)

    def test_create_item_with_parent(self):
        desktop_pidl = shell.SHGetSpecialFolderLocation(0, shellcon.CSIDL_DESKTOP)
        desktop_item = shell.SHCreateItemFromIDList(desktop_pidl, shell.IID_IShellItem)

        sf = shell.SHGetDesktopFolder()
        flags = shellcon.SHCONTF_FOLDERS | shellcon.SHCONTF_NONFOLDERS
        children = sf.EnumObjects(0, flags)
        child_pidl = next(children)
        item1 = shell.SHCreateItemWithParent(
            desktop_pidl, None, child_pidl, shell.IID_IShellItem
        )
        item2 = shell.SHCreateItemWithParent(None, sf, child_pidl, shell.IID_IShellItem)
        self.assertShellItemsEqual(item1, item2)


if __name__ == "__main__":
    unittest.main()
