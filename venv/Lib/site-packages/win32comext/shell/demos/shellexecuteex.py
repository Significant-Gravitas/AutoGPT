import win32con
from win32com.shell import shell, shellcon


def ExplorePIDL():
    pidl = shell.SHGetSpecialFolderLocation(0, shellcon.CSIDL_DESKTOP)
    print("The desktop is at", shell.SHGetPathFromIDList(pidl))
    shell.ShellExecuteEx(
        fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,
        nShow=win32con.SW_NORMAL,
        lpClass="folder",
        lpVerb="explore",
        lpIDList=pidl,
    )
    print("Done!")


if __name__ == "__main__":
    ExplorePIDL()
