# help.py - help utilities for PythonWin.
import os

import regutil
import win32api
import win32con
import win32ui

htmlhelp_handle = None

html_help_command_translators = {
    win32con.HELP_CONTENTS: 1,  # HH_DISPLAY_TOC
    win32con.HELP_CONTEXT: 15,  # HH_HELP_CONTEXT
    win32con.HELP_FINDER: 1,  # HH_DISPLAY_TOC
}


def FinalizeHelp():
    global htmlhelp_handle
    if htmlhelp_handle is not None:
        import win32help

        try:
            # frame = win32ui.GetMainFrame().GetSafeHwnd()
            frame = 0
            win32help.HtmlHelp(frame, None, win32help.HH_UNINITIALIZE, htmlhelp_handle)
        except win32help.error:
            print("Failed to finalize htmlhelp!")
        htmlhelp_handle = None


def OpenHelpFile(fileName, helpCmd=None, helpArg=None):
    "Open a help file, given a full path"
    # default help arg.
    win32ui.DoWaitCursor(1)
    try:
        if helpCmd is None:
            helpCmd = win32con.HELP_CONTENTS
        ext = os.path.splitext(fileName)[1].lower()
        if ext == ".hlp":
            win32api.WinHelp(
                win32ui.GetMainFrame().GetSafeHwnd(), fileName, helpCmd, helpArg
            )
        # XXX - using the htmlhelp API wreaks havoc with keyboard shortcuts
        # so we disable it, forcing ShellExecute, which works fine (but
        # doesn't close the help file when Pythonwin is closed.
        # Tom Heller also points out http://www.microsoft.com/mind/0499/faq/faq0499.asp,
        # which may or may not be related.
        elif 0 and ext == ".chm":
            import win32help

            global htmlhelp_handle
            helpCmd = html_help_command_translators.get(helpCmd, helpCmd)
            # frame = win32ui.GetMainFrame().GetSafeHwnd()
            frame = 0  # Dont want it overlapping ours!
            if htmlhelp_handle is None:
                htmlhelp_hwnd, htmlhelp_handle = win32help.HtmlHelp(
                    frame, None, win32help.HH_INITIALIZE
                )
            win32help.HtmlHelp(frame, fileName, helpCmd, helpArg)
        else:
            # Hope that the extension is registered, and we know what to do!
            win32api.ShellExecute(0, "open", fileName, None, "", win32con.SW_SHOW)
        return fileName
    finally:
        win32ui.DoWaitCursor(-1)


def ListAllHelpFiles():
    ret = []
    ret = _ListAllHelpFilesInRoot(win32con.HKEY_LOCAL_MACHINE)
    # Ensure we don't get dups.
    for item in _ListAllHelpFilesInRoot(win32con.HKEY_CURRENT_USER):
        if item not in ret:
            ret.append(item)
    return ret


def _ListAllHelpFilesInRoot(root):
    """Returns a list of (helpDesc, helpFname) for all registered help files"""
    import regutil

    retList = []
    try:
        key = win32api.RegOpenKey(
            root, regutil.BuildDefaultPythonKey() + "\\Help", 0, win32con.KEY_READ
        )
    except win32api.error as exc:
        import winerror

        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise
        return retList
    try:
        keyNo = 0
        while 1:
            try:
                helpDesc = win32api.RegEnumKey(key, keyNo)
                helpFile = win32api.RegQueryValue(key, helpDesc)
                retList.append((helpDesc, helpFile))
                keyNo = keyNo + 1
            except win32api.error as exc:
                import winerror

                if exc.winerror != winerror.ERROR_NO_MORE_ITEMS:
                    raise
                break
    finally:
        win32api.RegCloseKey(key)
    return retList


def SelectAndRunHelpFile():
    from pywin.dialogs import list

    helpFiles = ListAllHelpFiles()
    if len(helpFiles) == 1:
        # only 1 help file registered - probably ours - no point asking
        index = 0
    else:
        index = list.SelectFromLists("Select Help file", helpFiles, ["Title"])
    if index is not None:
        OpenHelpFile(helpFiles[index][1])


helpIDMap = None


def SetHelpMenuOtherHelp(mainMenu):
    """Modifies the main Help Menu to handle all registered help files.
    mainMenu -- The main menu to modify - usually from docTemplate.GetSharedMenu()
    """

    # Load all help files from the registry.
    global helpIDMap
    if helpIDMap is None:
        helpIDMap = {}
        cmdID = win32ui.ID_HELP_OTHER
        excludeList = ["Main Python Documentation", "Pythonwin Reference"]
        firstList = ListAllHelpFiles()
        # We actually want to not only exclude these entries, but
        # their help file names (as many entries may share the same name)
        excludeFnames = []
        for desc, fname in firstList:
            if desc in excludeList:
                excludeFnames.append(fname)

        helpDescs = []
        for desc, fname in firstList:
            if fname not in excludeFnames:
                helpIDMap[cmdID] = (desc, fname)
                win32ui.GetMainFrame().HookCommand(HandleHelpOtherCommand, cmdID)
                cmdID = cmdID + 1

    helpMenu = mainMenu.GetSubMenu(
        mainMenu.GetMenuItemCount() - 1
    )  # Help menu always last.
    otherHelpMenuPos = 2  # cant search for ID, as sub-menu has no ID.
    otherMenu = helpMenu.GetSubMenu(otherHelpMenuPos)
    while otherMenu.GetMenuItemCount():
        otherMenu.DeleteMenu(0, win32con.MF_BYPOSITION)

    if helpIDMap:
        for id, (desc, fname) in helpIDMap.items():
            otherMenu.AppendMenu(win32con.MF_ENABLED | win32con.MF_STRING, id, desc)
    else:
        helpMenu.EnableMenuItem(
            otherHelpMenuPos, win32con.MF_BYPOSITION | win32con.MF_GRAYED
        )


def HandleHelpOtherCommand(cmd, code):
    OpenHelpFile(helpIDMap[cmd][1])
