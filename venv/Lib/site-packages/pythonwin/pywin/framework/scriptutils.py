"""
Various utilities for running/importing a script
"""
import bdb
import linecache
import os
import sys
import traceback

import __main__
import win32api
import win32con
import win32ui
from pywin.mfc import dialog
from pywin.mfc.docview import TreeView

from .cmdline import ParseArgs

RS_DEBUGGER_NONE = 0  # Dont run under the debugger.
RS_DEBUGGER_STEP = 1  # Start stepping under the debugger
RS_DEBUGGER_GO = 2  # Just run under the debugger, stopping only at break-points.
RS_DEBUGGER_PM = 3  # Dont run under debugger, but do post-mortem analysis on exception.

debugging_options = """No debugging
Step-through in the debugger
Run in the debugger
Post-Mortem of unhandled exceptions""".split(
    "\n"
)

byte_cr = "\r".encode("ascii")
byte_lf = "\n".encode("ascii")
byte_crlf = "\r\n".encode("ascii")


# A dialog box for the "Run Script" command.
class DlgRunScript(dialog.Dialog):
    "A class for the 'run script' dialog"

    def __init__(self, bHaveDebugger):
        dialog.Dialog.__init__(self, win32ui.IDD_RUN_SCRIPT)
        self.AddDDX(win32ui.IDC_EDIT1, "script")
        self.AddDDX(win32ui.IDC_EDIT2, "args")
        self.AddDDX(win32ui.IDC_COMBO1, "debuggingType", "i")
        self.HookCommand(self.OnBrowse, win32ui.IDC_BUTTON2)
        self.bHaveDebugger = bHaveDebugger

    def OnInitDialog(self):
        rc = dialog.Dialog.OnInitDialog(self)
        cbo = self.GetDlgItem(win32ui.IDC_COMBO1)
        for o in debugging_options:
            cbo.AddString(o)
        cbo.SetCurSel(self["debuggingType"])
        if not self.bHaveDebugger:
            cbo.EnableWindow(0)

    def OnBrowse(self, id, code):
        if code != 0:  # BN_CLICKED
            return 1
        openFlags = win32con.OFN_OVERWRITEPROMPT | win32con.OFN_FILEMUSTEXIST
        dlg = win32ui.CreateFileDialog(
            1, None, None, openFlags, "Python Scripts (*.py)|*.py||", self
        )
        dlg.SetOFNTitle("Run Script")
        if dlg.DoModal() != win32con.IDOK:
            return 0
        self["script"] = dlg.GetPathName()
        self.UpdateData(0)
        return 0


def GetDebugger():
    """Get the default Python debugger.  Returns the debugger, or None.

    It is assumed the debugger has a standard "pdb" defined interface.
    Currently always returns the 'pywin.debugger' debugger, or None
    (pdb is _not_ returned as it is not effective in this GUI environment)
    """
    try:
        import pywin.debugger

        return pywin.debugger
    except ImportError:
        return None


def IsOnPythonPath(path):
    "Given a path only, see if it is on the Pythonpath.  Assumes path is a full path spec."
    # must check that the command line arg's path is in sys.path
    for syspath in sys.path:
        try:
            # Python 1.5 and later allows an empty sys.path entry.
            if syspath and win32ui.FullPath(syspath) == path:
                return 1
        except win32ui.error as details:
            print(
                "Warning: The sys.path entry '%s' is invalid\n%s" % (syspath, details)
            )
    return 0


def GetPackageModuleName(fileName):
    """Given a filename, return (module name, new path).
    eg - given "c:\a\b\c\my.py", return ("b.c.my",None) if "c:\a" is on sys.path.
    If no package found, will return ("my", "c:\a\b\c")
    """
    path, fname = os.path.split(fileName)
    path = origPath = win32ui.FullPath(path)
    fname = os.path.splitext(fname)[0]
    modBits = []
    newPathReturn = None
    if not IsOnPythonPath(path):
        # Module not directly on the search path - see if under a package.
        while len(path) > 3:  # ie 'C:\'
            path, modBit = os.path.split(path)
            modBits.append(modBit)
            # If on path, _and_ existing package of that name loaded.
            if (
                IsOnPythonPath(path)
                and modBit in sys.modules
                and (
                    os.path.exists(os.path.join(path, modBit, "__init__.py"))
                    or os.path.exists(os.path.join(path, modBit, "__init__.pyc"))
                    or os.path.exists(os.path.join(path, modBit, "__init__.pyo"))
                )
            ):
                modBits.reverse()
                return ".".join(modBits) + "." + fname, newPathReturn
            # Not found - look a level higher
        else:
            newPathReturn = origPath

    return fname, newPathReturn


def GetActiveView():
    """Gets the edit control (eg, EditView) with the focus, or None"""
    try:
        childFrame, bIsMaximised = win32ui.GetMainFrame().MDIGetActive()
        return childFrame.GetActiveView()
    except win32ui.error:
        return None


def GetActiveEditControl():
    view = GetActiveView()
    if view is None:
        return None
    if hasattr(view, "SCIAddText"):  # Is it a scintilla control?
        return view
    try:
        return view.GetRichEditCtrl()
    except AttributeError:
        pass
    try:
        return view.GetEditCtrl()
    except AttributeError:
        pass


def GetActiveEditorDocument():
    """Returns the active editor document and view, or (None,None) if no
    active document or its not an editor document.
    """
    view = GetActiveView()
    if view is None or isinstance(view, TreeView):
        return (None, None)
    doc = view.GetDocument()
    if hasattr(doc, "MarkerAdd"):  # Is it an Editor document?
        return doc, view
    return (None, None)


def GetActiveFileName(bAutoSave=1):
    """Gets the file name for the active frame, saving it if necessary.

    Returns None if it cant be found, or raises KeyboardInterrupt.
    """
    pathName = None
    active = GetActiveView()
    if active is None:
        return None
    try:
        doc = active.GetDocument()
        pathName = doc.GetPathName()

        if bAutoSave and (
            len(pathName) > 0
            or doc.GetTitle()[:8] == "Untitled"
            or doc.GetTitle()[:6] == "Script"
        ):  # if not a special purpose window
            if doc.IsModified():
                try:
                    doc.OnSaveDocument(pathName)
                    pathName = doc.GetPathName()

                    # clear the linecache buffer
                    linecache.clearcache()

                except win32ui.error:
                    raise KeyboardInterrupt

    except (win32ui.error, AttributeError):
        pass
    if not pathName:
        return None
    return pathName


lastScript = ""
lastArgs = ""
lastDebuggingType = RS_DEBUGGER_NONE


def RunScript(defName=None, defArgs=None, bShowDialog=1, debuggingType=None):
    global lastScript, lastArgs, lastDebuggingType
    _debugger_stop_frame_ = 1  # Magic variable so the debugger will hide me!

    # Get the debugger - may be None!
    debugger = GetDebugger()

    if defName is None:
        try:
            pathName = GetActiveFileName()
        except KeyboardInterrupt:
            return  # User cancelled save.
    else:
        pathName = defName
    if not pathName:
        pathName = lastScript
    if defArgs is None:
        args = ""
        if pathName == lastScript:
            args = lastArgs
    else:
        args = defArgs
    if debuggingType is None:
        debuggingType = lastDebuggingType

    if not pathName or bShowDialog:
        dlg = DlgRunScript(debugger is not None)
        dlg["script"] = pathName
        dlg["args"] = args
        dlg["debuggingType"] = debuggingType
        if dlg.DoModal() != win32con.IDOK:
            return
        script = dlg["script"]
        args = dlg["args"]
        debuggingType = dlg["debuggingType"]
        if not script:
            return
        if debuggingType == RS_DEBUGGER_GO and debugger is not None:
            # This may surprise users - they select "Run under debugger", but
            # it appears not to!  Only warn when they pick from the dialog!
            # First - ensure the debugger is activated to pickup any break-points
            # set in the editor.
            try:
                # Create the debugger, but _dont_ init the debugger GUI.
                rd = debugger._GetCurrentDebugger()
            except AttributeError:
                rd = None
            if rd is not None and len(rd.breaks) == 0:
                msg = "There are no active break-points.\r\n\r\nSelecting this debug option without any\r\nbreak-points is unlikely to have the desired effect\r\nas the debugger is unlikely to be invoked..\r\n\r\nWould you like to step-through in the debugger instead?"
                rc = win32ui.MessageBox(
                    msg,
                    win32ui.LoadString(win32ui.IDR_DEBUGGER),
                    win32con.MB_YESNOCANCEL | win32con.MB_ICONINFORMATION,
                )
                if rc == win32con.IDCANCEL:
                    return
                if rc == win32con.IDYES:
                    debuggingType = RS_DEBUGGER_STEP

        lastDebuggingType = debuggingType
        lastScript = script
        lastArgs = args
    else:
        script = pathName

    # try and open the script.
    if (
        len(os.path.splitext(script)[1]) == 0
    ):  # check if no extension supplied, and give one.
        script = script + ".py"
    # If no path specified, try and locate the file
    path, fnameonly = os.path.split(script)
    if len(path) == 0:
        try:
            os.stat(fnameonly)  # See if it is OK as is...
            script = fnameonly
        except os.error:
            fullScript = LocatePythonFile(script)
            if fullScript is None:
                win32ui.MessageBox("The file '%s' can not be located" % script)
                return
            script = fullScript
    else:
        path = win32ui.FullPath(path)
        if not IsOnPythonPath(path):
            sys.path.append(path)

    # py3k fun: If we use text mode to open the file, we get \r\n
    # translated so Python allows the syntax (good!), but we get back
    # text already decoded from the default encoding (bad!) and Python
    # ignores any encoding decls (bad!).  If we use binary mode we get
    # the raw bytes and Python looks at the encoding (good!) but \r\n
    # chars stay in place so Python throws a syntax error (bad!).
    # So: so the binary thing and manually normalize \r\n.
    try:
        f = open(script, "rb")
    except IOError as exc:
        win32ui.MessageBox(
            "The file could not be opened - %s (%d)" % (exc.strerror, exc.errno)
        )
        return

    # Get the source-code - as above, normalize \r\n
    code = f.read().replace(byte_crlf, byte_lf).replace(byte_cr, byte_lf) + byte_lf

    # Remember and hack sys.argv for the script.
    oldArgv = sys.argv
    sys.argv = ParseArgs(args)
    sys.argv.insert(0, script)
    # sys.path[0] is the path of the script
    oldPath0 = sys.path[0]
    newPath0 = os.path.split(script)[0]
    if not oldPath0:  # if sys.path[0] is empty
        sys.path[0] = newPath0
        insertedPath0 = 0
    else:
        sys.path.insert(0, newPath0)
        insertedPath0 = 1
    bWorked = 0
    win32ui.DoWaitCursor(1)
    base = os.path.split(script)[1]
    # Allow windows to repaint before starting.
    win32ui.PumpWaitingMessages()
    win32ui.SetStatusText("Running script %s..." % base, 1)
    exitCode = 0
    from pywin.framework import interact

    # Check the debugger flags
    if debugger is None and (debuggingType != RS_DEBUGGER_NONE):
        win32ui.MessageBox(
            "No debugger is installed.  Debugging options have been ignored!"
        )
        debuggingType = RS_DEBUGGER_NONE

    # Get a code object - ignore the debugger for this, as it is probably a syntax error
    # at this point
    try:
        codeObject = compile(code, script, "exec")
    except:
        # Almost certainly a syntax error!
        _HandlePythonFailure("run script", script)
        # No code object which to run/debug.
        return
    __main__.__file__ = script
    try:
        if debuggingType == RS_DEBUGGER_STEP:
            debugger.run(codeObject, __main__.__dict__, start_stepping=1)
        elif debuggingType == RS_DEBUGGER_GO:
            debugger.run(codeObject, __main__.__dict__, start_stepping=0)
        else:
            # Post mortem or no debugging
            exec(codeObject, __main__.__dict__)
        bWorked = 1
    except bdb.BdbQuit:
        # Dont print tracebacks when the debugger quit, but do print a message.
        print("Debugging session cancelled.")
        exitCode = 1
        bWorked = 1
    except SystemExit as code:
        exitCode = code
        bWorked = 1
    except KeyboardInterrupt:
        # Consider this successful, as we dont want the debugger.
        # (but we do want a traceback!)
        if interact.edit and interact.edit.currentView:
            interact.edit.currentView.EnsureNoPrompt()
        traceback.print_exc()
        if interact.edit and interact.edit.currentView:
            interact.edit.currentView.AppendToPrompt([])
        bWorked = 1
    except:
        if interact.edit and interact.edit.currentView:
            interact.edit.currentView.EnsureNoPrompt()
        traceback.print_exc()
        if interact.edit and interact.edit.currentView:
            interact.edit.currentView.AppendToPrompt([])
        if debuggingType == RS_DEBUGGER_PM:
            debugger.pm()
    del __main__.__file__
    sys.argv = oldArgv
    if insertedPath0:
        del sys.path[0]
    else:
        sys.path[0] = oldPath0
    f.close()
    if bWorked:
        win32ui.SetStatusText("Script '%s' returned exit code %s" % (script, exitCode))
    else:
        win32ui.SetStatusText("Exception raised while running script  %s" % base)
    try:
        sys.stdout.flush()
    except AttributeError:
        pass

    win32ui.DoWaitCursor(0)


def ImportFile():
    """This code looks for the current window, and determines if it can be imported.  If not,
    it will prompt for a file name, and allow it to be imported."""
    try:
        pathName = GetActiveFileName()
    except KeyboardInterrupt:
        pathName = None

    if pathName is not None:
        if os.path.splitext(pathName)[1].lower() not in (".py", ".pyw", ".pyx"):
            pathName = None

    if pathName is None:
        openFlags = win32con.OFN_OVERWRITEPROMPT | win32con.OFN_FILEMUSTEXIST
        dlg = win32ui.CreateFileDialog(
            1, None, None, openFlags, "Python Scripts (*.py;*.pyw)|*.py;*.pyw;*.pyx||"
        )
        dlg.SetOFNTitle("Import Script")
        if dlg.DoModal() != win32con.IDOK:
            return 0

        pathName = dlg.GetPathName()

    # If already imported, dont look for package
    path, modName = os.path.split(pathName)
    modName, modExt = os.path.splitext(modName)
    newPath = None
    # note that some packages (*cough* email *cough*) use "lazy importers"
    # meaning sys.modules can change as a side-effect of looking at
    # module.__file__ - so we must take a copy (ie, items() in py2k,
    # list(items()) in py3k)
    for key, mod in list(sys.modules.items()):
        if getattr(mod, "__file__", None):
            fname = mod.__file__
            base, ext = os.path.splitext(fname)
            if ext.lower() in (".pyo", ".pyc"):
                ext = ".py"
            fname = base + ext
            if win32ui.ComparePath(fname, pathName):
                modName = key
                break
    else:  # for not broken
        modName, newPath = GetPackageModuleName(pathName)
        if newPath:
            sys.path.append(newPath)

    if modName in sys.modules:
        bNeedReload = 1
        what = "reload"
    else:
        what = "import"
        bNeedReload = 0

    win32ui.SetStatusText(what.capitalize() + "ing module...", 1)
    win32ui.DoWaitCursor(1)
    # 	win32ui.GetMainFrame().BeginWaitCursor()

    try:
        # always do an import, as it is cheap if it's already loaded.  This ensures
        # it is in our name space.
        codeObj = compile("import " + modName, "<auto import>", "exec")
    except SyntaxError:
        win32ui.SetStatusText('Invalid filename for import: "' + modName + '"')
        return
    try:
        exec(codeObj, __main__.__dict__)
        mod = sys.modules.get(modName)
        if bNeedReload:
            from importlib import reload

            mod = reload(sys.modules[modName])
        win32ui.SetStatusText(
            "Successfully "
            + what
            + "ed module '"
            + modName
            + "': %s" % getattr(mod, "__file__", "<unkown file>")
        )
    except:
        _HandlePythonFailure(what)
    win32ui.DoWaitCursor(0)


def CheckFile():
    """This code looks for the current window, and gets Python to check it
    without actually executing any code (ie, by compiling only)
    """
    try:
        pathName = GetActiveFileName()
    except KeyboardInterrupt:
        return

    what = "check"
    win32ui.SetStatusText(what.capitalize() + "ing module...", 1)
    win32ui.DoWaitCursor(1)
    try:
        f = open(pathName)
    except IOError as details:
        print("Cant open file '%s' - %s" % (pathName, details))
        return
    try:
        code = f.read() + "\n"
    finally:
        f.close()
    try:
        codeObj = compile(code, pathName, "exec")
        if RunTabNanny(pathName):
            win32ui.SetStatusText(
                "Python and the TabNanny successfully checked the file '"
                + os.path.basename(pathName)
                + "'"
            )
    except SyntaxError:
        _HandlePythonFailure(what, pathName)
    except:
        traceback.print_exc()
        _HandlePythonFailure(what)
    win32ui.DoWaitCursor(0)


def RunTabNanny(filename):
    import io as io

    tabnanny = FindTabNanny()
    if tabnanny is None:
        win32ui.MessageBox("The TabNanny is not around, so the children can run amok!")
        return

    # Capture the tab-nanny output
    newout = io.StringIO()
    old_out = sys.stderr, sys.stdout
    sys.stderr = sys.stdout = newout
    try:
        tabnanny.check(filename)
    finally:
        # Restore output
        sys.stderr, sys.stdout = old_out
    data = newout.getvalue()
    if data:
        try:
            lineno = data.split()[1]
            lineno = int(lineno)
            _JumpToPosition(filename, lineno)
            try:  # Try and display whitespace
                GetActiveEditControl().SCISetViewWS(1)
            except:
                pass
            win32ui.SetStatusText("The TabNanny found trouble at line %d" % lineno)
        except (IndexError, TypeError, ValueError):
            print("The tab nanny complained, but I cant see where!")
            print(data)
        return 0
    return 1


def _JumpToPosition(fileName, lineno, col=1):
    JumpToDocument(fileName, lineno, col)


def JumpToDocument(fileName, lineno=0, col=1, nChars=0, bScrollToTop=0):
    # Jump to the position in a file.
    # If lineno is <= 0, dont move the position - just open/restore.
    # if nChars > 0, select that many characters.
    # if bScrollToTop, the specified line will be moved to the top of the window
    #  (eg, bScrollToTop should be false when jumping to an error line to retain the
    #  context, but true when jumping to a method defn, where we want the full body.
    # Return the view which is editing the file, or None on error.
    doc = win32ui.GetApp().OpenDocumentFile(fileName)
    if doc is None:
        return None
    frame = doc.GetFirstView().GetParentFrame()
    try:
        view = frame.GetEditorView()
        if frame.GetActiveView() != view:
            frame.SetActiveView(view)
        frame.AutoRestore()
    except AttributeError:  # Not an editor frame??
        view = doc.GetFirstView()
    if lineno > 0:
        charNo = view.LineIndex(lineno - 1)
        start = charNo + col - 1
        size = view.GetTextLength()
        try:
            view.EnsureCharsVisible(charNo)
        except AttributeError:
            print("Doesnt appear to be one of our views?")
        view.SetSel(min(start, size), min(start + nChars, size))
    if bScrollToTop:
        curTop = view.GetFirstVisibleLine()
        nScroll = (lineno - 1) - curTop
        view.LineScroll(nScroll, 0)
    view.SetFocus()
    return view


def _HandlePythonFailure(what, syntaxErrorPathName=None):
    typ, details, tb = sys.exc_info()
    if isinstance(details, SyntaxError):
        try:
            msg, (fileName, line, col, text) = details
            if (not fileName or fileName == "<string>") and syntaxErrorPathName:
                fileName = syntaxErrorPathName
            _JumpToPosition(fileName, line, col)
        except (TypeError, ValueError):
            msg = str(details)
        win32ui.SetStatusText("Failed to " + what + " - syntax error - %s" % msg)
    else:
        traceback.print_exc()
        win32ui.SetStatusText("Failed to " + what + " - " + str(details))
    tb = None  # Clean up a cycle.


# Find the Python TabNanny in either the standard library or the Python Tools/Scripts directory.
def FindTabNanny():
    try:
        return __import__("tabnanny")
    except ImportError:
        pass
    # OK - not in the standard library - go looking.
    filename = "tabnanny.py"
    try:
        path = win32api.RegQueryValue(
            win32con.HKEY_LOCAL_MACHINE,
            "SOFTWARE\\Python\\PythonCore\\%s\\InstallPath" % (sys.winver),
        )
    except win32api.error:
        print("WARNING - The Python registry does not have an 'InstallPath' setting")
        print("          The file '%s' can not be located" % (filename))
        return None
    fname = os.path.join(path, "Tools\\Scripts\\%s" % filename)
    try:
        os.stat(fname)
    except os.error:
        print(
            "WARNING - The file '%s' can not be located in path '%s'" % (filename, path)
        )
        return None

    tabnannyhome, tabnannybase = os.path.split(fname)
    tabnannybase = os.path.splitext(tabnannybase)[0]
    # Put tab nanny at the top of the path.
    sys.path.insert(0, tabnannyhome)
    try:
        return __import__(tabnannybase)
    finally:
        # remove the tab-nanny from the path
        del sys.path[0]


def LocatePythonFile(fileName, bBrowseIfDir=1):
    "Given a file name, return a fully qualified file name, or None"
    # first look for the exact file as specified
    if not os.path.isfile(fileName):
        # Go looking!
        baseName = fileName
        for path in sys.path:
            fileName = os.path.abspath(os.path.join(path, baseName))
            if os.path.isdir(fileName):
                if bBrowseIfDir:
                    d = win32ui.CreateFileDialog(
                        1, "*.py", None, 0, "Python Files (*.py)|*.py|All files|*.*"
                    )
                    d.SetOFNInitialDir(fileName)
                    rc = d.DoModal()
                    if rc == win32con.IDOK:
                        fileName = d.GetPathName()
                        break
                    else:
                        return None
            else:
                fileName = fileName + ".py"
                if os.path.isfile(fileName):
                    break  # Found it!

        else:  # for not broken out of
            return None
    return win32ui.FullPath(fileName)
