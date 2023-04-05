# intpyapp.py  - Interactive Python application class
#
import os
import sys
import traceback

import __main__
import commctrl
import win32api
import win32con
import win32ui
from pywin.mfc import afxres, dialog

from . import app, dbgcommands

lastLocateFileName = ".py"  # used in the "File/Locate" dialog...


# todo - _SetupSharedMenu should be moved to a framework class.
def _SetupSharedMenu_(self):
    sharedMenu = self.GetSharedMenu()
    from pywin.framework import toolmenu

    toolmenu.SetToolsMenu(sharedMenu)
    from pywin.framework import help

    help.SetHelpMenuOtherHelp(sharedMenu)


from pywin.mfc import docview

docview.DocTemplate._SetupSharedMenu_ = _SetupSharedMenu_


class MainFrame(app.MainFrame):
    def OnCreate(self, createStruct):
        self.closing = 0
        if app.MainFrame.OnCreate(self, createStruct) == -1:
            return -1
        style = (
            win32con.WS_CHILD
            | afxres.CBRS_SIZE_DYNAMIC
            | afxres.CBRS_TOP
            | afxres.CBRS_TOOLTIPS
            | afxres.CBRS_FLYBY
        )

        self.EnableDocking(afxres.CBRS_ALIGN_ANY)

        tb = win32ui.CreateToolBar(self, style | win32con.WS_VISIBLE)
        tb.ModifyStyle(0, commctrl.TBSTYLE_FLAT)
        tb.LoadToolBar(win32ui.IDR_MAINFRAME)
        tb.EnableDocking(afxres.CBRS_ALIGN_ANY)
        tb.SetWindowText("Standard")
        self.DockControlBar(tb)
        # Any other packages which use toolbars
        from pywin.debugger.debugger import PrepareControlBars

        PrepareControlBars(self)
        # Note "interact" also uses dockable windows, but they already happen

        # And a "Tools" menu on the main frame.
        menu = self.GetMenu()
        from . import toolmenu

        toolmenu.SetToolsMenu(menu, 2)
        # And fix the "Help" menu on the main frame
        from pywin.framework import help

        help.SetHelpMenuOtherHelp(menu)

    def OnClose(self):
        try:
            import pywin.debugger

            if (
                pywin.debugger.currentDebugger is not None
                and pywin.debugger.currentDebugger.pumping
            ):
                try:
                    pywin.debugger.currentDebugger.close(1)
                except:
                    traceback.print_exc()
                return
        except win32ui.error:
            pass
        self.closing = 1
        self.SaveBarState("ToolbarDefault")
        self.SetActiveView(None)  # Otherwise MFC's OnClose may _not_ prompt for save.

        from pywin.framework import help

        help.FinalizeHelp()

        self.DestroyControlBar(afxres.AFX_IDW_TOOLBAR)
        self.DestroyControlBar(win32ui.ID_VIEW_TOOLBAR_DBG)

        return self._obj_.OnClose()

    def DestroyControlBar(self, id):
        try:
            bar = self.GetControlBar(id)
        except win32ui.error:
            return
        bar.DestroyWindow()

    def OnCommand(self, wparam, lparam):
        # By default, the current MDI child frame will process WM_COMMAND
        # messages before any docked control bars - even if the control bar
        # has focus.  This is a problem for the interactive window when docked.
        # Therefore, we detect the situation of a view having the main frame
        # as its parent, and assume it must be a docked view (which it will in an MDI app)
        try:
            v = (
                self.GetActiveView()
            )  # Raise an exception if none - good - then we want default handling
            # Main frame _does_ have a current view (ie, a docking view) - see if it wants it.
            if v.OnCommand(wparam, lparam):
                return 1
        except (win32ui.error, AttributeError):
            pass
        return self._obj_.OnCommand(wparam, lparam)


class InteractivePythonApp(app.CApp):
    # This works if necessary - just we dont need to override the Run method.
    # 	def Run(self):
    # 		return self._obj_.Run()

    def HookCommands(self):
        app.CApp.HookCommands(self)
        dbgcommands.DebuggerCommandHandler().HookCommands()
        self.HookCommand(self.OnViewBrowse, win32ui.ID_VIEW_BROWSE)
        self.HookCommand(self.OnFileImport, win32ui.ID_FILE_IMPORT)
        self.HookCommand(self.OnFileCheck, win32ui.ID_FILE_CHECK)
        self.HookCommandUpdate(self.OnUpdateFileCheck, win32ui.ID_FILE_CHECK)
        self.HookCommand(self.OnFileRun, win32ui.ID_FILE_RUN)
        self.HookCommand(self.OnFileLocate, win32ui.ID_FILE_LOCATE)
        self.HookCommand(self.OnInteractiveWindow, win32ui.ID_VIEW_INTERACTIVE)
        self.HookCommandUpdate(
            self.OnUpdateInteractiveWindow, win32ui.ID_VIEW_INTERACTIVE
        )
        self.HookCommand(self.OnViewOptions, win32ui.ID_VIEW_OPTIONS)
        self.HookCommand(self.OnHelpIndex, afxres.ID_HELP_INDEX)
        self.HookCommand(self.OnFileSaveAll, win32ui.ID_FILE_SAVE_ALL)
        self.HookCommand(self.OnViewToolbarDbg, win32ui.ID_VIEW_TOOLBAR_DBG)
        self.HookCommandUpdate(self.OnUpdateViewToolbarDbg, win32ui.ID_VIEW_TOOLBAR_DBG)

    def CreateMainFrame(self):
        return MainFrame()

    def MakeExistingDDEConnection(self):
        # Use DDE to connect to an existing instance
        # Return None if no existing instance
        try:
            from . import intpydde
        except ImportError:
            # No dde support!
            return None
        conv = intpydde.CreateConversation(self.ddeServer)
        try:
            conv.ConnectTo("Pythonwin", "System")
            return conv
        except intpydde.error:
            return None

    def InitDDE(self):
        # Do all the magic DDE handling.
        # Returns TRUE if we have pumped the arguments to our
        # remote DDE app, and we should terminate.
        try:
            from . import intpydde
        except ImportError:
            self.ddeServer = None
            intpydde = None
        if intpydde is not None:
            self.ddeServer = intpydde.DDEServer(self)
            self.ddeServer.Create("Pythonwin", intpydde.CBF_FAIL_SELFCONNECTIONS)
            try:
                # If there is an existing instance, pump the arguments to it.
                connection = self.MakeExistingDDEConnection()
                if connection is not None:
                    connection.Exec("self.Activate()")
                    if self.ProcessArgs(sys.argv, connection) is None:
                        return 1
            except:
                # It is too early to 'print' an exception - we
                # don't have stdout setup yet!
                win32ui.DisplayTraceback(
                    sys.exc_info(), " - error in DDE conversation with Pythonwin"
                )
                return 1

    def InitInstance(self):
        # Allow "/nodde" and "/new" to optimize this!
        if (
            "/nodde" not in sys.argv
            and "/new" not in sys.argv
            and "-nodde" not in sys.argv
            and "-new" not in sys.argv
        ):
            if self.InitDDE():
                return 1  # A remote DDE client is doing it for us!
        else:
            self.ddeServer = None

        win32ui.SetRegistryKey(
            "Python %s" % (sys.winver,)
        )  # MFC automatically puts the main frame caption on!
        app.CApp.InitInstance(self)

        # Create the taskbar icon
        win32ui.CreateDebuggerThread()

        # Allow Pythonwin to host OCX controls.
        win32ui.EnableControlContainer()

        # Display the interactive window if the user wants it.
        from . import interact

        interact.CreateInteractiveWindowUserPreference()

        # Load the modules we use internally.
        self.LoadSystemModules()

        # Load additional module the user may want.
        self.LoadUserModules()

        # Load the ToolBar state near the end of the init process, as
        # there may be Toolbar IDs created by the user or other modules.
        # By now all these modules should be loaded, so all the toolbar IDs loaded.
        try:
            self.frame.LoadBarState("ToolbarDefault")
        except win32ui.error:
            # MFC sucks.  It does essentially "GetDlgItem(x)->Something", so if the
            # toolbar with ID x does not exist, MFC crashes!  Pythonwin has a trap for this
            # but I need to investigate more how to prevent it (AFAIK, ensuring all the
            # toolbars are created by now _should_ stop it!)
            pass

        # Finally process the command line arguments.
        try:
            self.ProcessArgs(sys.argv)
        except:
            # too early for printing anything.
            win32ui.DisplayTraceback(
                sys.exc_info(), " - error processing command line args"
            )

    def ExitInstance(self):
        win32ui.DestroyDebuggerThread()
        try:
            from . import interact

            interact.DestroyInteractiveWindow()
        except:
            pass
        if self.ddeServer is not None:
            self.ddeServer.Shutdown()
            self.ddeServer = None
        return app.CApp.ExitInstance(self)

    def Activate(self):
        # Bring to the foreground.  Mainly used when another app starts up, it asks
        # this one to activate itself, then it terminates.
        frame = win32ui.GetMainFrame()
        frame.SetForegroundWindow()
        if frame.GetWindowPlacement()[1] == win32con.SW_SHOWMINIMIZED:
            frame.ShowWindow(win32con.SW_RESTORE)

    def ProcessArgs(self, args, dde=None):
        # If we are going to talk to a remote app via DDE, then
        # activate it!
        if (
            len(args) < 1 or not args[0]
        ):  # argv[0]=='' when started without args, just like Python.exe!
            return

        i = 0
        while i < len(args):
            argType = args[i]
            i += 1
            if argType.startswith("-"):
                # Support dash options. Slash options are misinterpreted by python init
                # as path and not finding usually 'C:\\' ends up in sys.path[0]
                argType = "/" + argType[1:]
            if not argType.startswith("/"):
                argType = win32ui.GetProfileVal(
                    "Python", "Default Arg Type", "/edit"
                ).lower()
                i -= 1  #  arg is /edit's parameter
            par = i < len(args) and args[i] or "MISSING"
            if argType in ("/nodde", "/new", "-nodde", "-new"):
                # Already handled
                pass
            elif argType.startswith("/goto:"):
                gotoline = int(argType[len("/goto:") :])
                if dde:
                    dde.Exec(
                        "from pywin.framework import scriptutils\n"
                        "ed = scriptutils.GetActiveEditControl()\n"
                        "if ed: ed.SetSel(ed.LineIndex(%s - 1))" % gotoline
                    )
                else:
                    from . import scriptutils

                    ed = scriptutils.GetActiveEditControl()
                    if ed:
                        ed.SetSel(ed.LineIndex(gotoline - 1))
            elif argType == "/edit":
                # Load up the default application.
                i += 1
                fname = win32api.GetFullPathName(par)
                if not os.path.isfile(fname):
                    # if we don't catch this, OpenDocumentFile() (actually
                    # PyCDocument.SetPathName() in
                    # pywin.scintilla.document.CScintillaDocument.OnOpenDocument)
                    # segfaults Pythonwin on recent PY3 builds (b228)
                    win32ui.MessageBox(
                        "No such file: %s\n\nCommand Line: %s"
                        % (fname, win32api.GetCommandLine()),
                        "Open file for edit",
                        win32con.MB_ICONERROR,
                    )
                    continue
                if dde:
                    dde.Exec("win32ui.GetApp().OpenDocumentFile(%s)" % (repr(fname)))
                else:
                    win32ui.GetApp().OpenDocumentFile(par)
            elif argType == "/rundlg":
                if dde:
                    dde.Exec(
                        "from pywin.framework import scriptutils;scriptutils.RunScript(%r, %r, 1)"
                        % (par, " ".join(args[i + 1 :]))
                    )
                else:
                    from . import scriptutils

                    scriptutils.RunScript(par, " ".join(args[i + 1 :]))
                return
            elif argType == "/run":
                if dde:
                    dde.Exec(
                        "from pywin.framework import scriptutils;scriptutils.RunScript(%r, %r, 0)"
                        % (par, " ".join(args[i + 1 :]))
                    )
                else:
                    from . import scriptutils

                    scriptutils.RunScript(par, " ".join(args[i + 1 :]), 0)
                return
            elif argType == "/app":
                raise RuntimeError(
                    "/app only supported for new instances of Pythonwin.exe"
                )
            elif argType == "/dde":  # Send arbitary command
                if dde is not None:
                    dde.Exec(par)
                else:
                    win32ui.MessageBox(
                        "The /dde command can only be used\r\nwhen Pythonwin is already running"
                    )
                i += 1
            else:
                raise ValueError("Command line argument not recognised: %s" % argType)

    def LoadSystemModules(self):
        self.DoLoadModules("pywin.framework.editor,pywin.framework.stdin")

    def LoadUserModules(self, moduleNames=None):
        # Load the users modules.
        if moduleNames is None:
            default = "pywin.framework.sgrepmdi,pywin.framework.mdi_pychecker"
            moduleNames = win32ui.GetProfileVal("Python", "Startup Modules", default)
        self.DoLoadModules(moduleNames)

    def DoLoadModules(self, moduleNames):  # ", sep string of module names.
        if not moduleNames:
            return
        modules = moduleNames.split(",")
        for module in modules:
            try:
                __import__(module)
            except:  # Catch em all, else the app itself dies! 'ImportError:
                traceback.print_exc()
                msg = 'Startup import of user module "%s" failed' % module
                print(msg)
                win32ui.MessageBox(msg)

    #
    # DDE Callback
    #
    def OnDDECommand(self, command):
        try:
            exec(command + "\n")
        except:
            print("ERROR executing DDE command: ", command)
            traceback.print_exc()
            raise

    #
    # General handlers
    #
    def OnViewBrowse(self, id, code):
        "Called when ViewBrowse message is received"
        from pywin.tools import browser

        obName = dialog.GetSimpleInput("Object", "__builtins__", "Browse Python Object")
        if obName is None:
            return
        try:
            browser.Browse(eval(obName, __main__.__dict__, __main__.__dict__))
        except NameError:
            win32ui.MessageBox("This is no object with this name")
        except AttributeError:
            win32ui.MessageBox("The object has no attribute of that name")
        except:
            traceback.print_exc()
            win32ui.MessageBox("This object can not be browsed")

    def OnFileImport(self, id, code):
        "Called when a FileImport message is received. Import the current or specified file"
        from . import scriptutils

        scriptutils.ImportFile()

    def OnFileCheck(self, id, code):
        "Called when a FileCheck message is received. Check the current file."
        from . import scriptutils

        scriptutils.CheckFile()

    def OnUpdateFileCheck(self, cmdui):
        from . import scriptutils

        cmdui.Enable(scriptutils.GetActiveFileName(0) is not None)

    def OnFileRun(self, id, code):
        "Called when a FileRun message is received."
        from . import scriptutils

        showDlg = win32api.GetKeyState(win32con.VK_SHIFT) >= 0
        scriptutils.RunScript(None, None, showDlg)

    def OnFileLocate(self, id, code):
        from . import scriptutils

        global lastLocateFileName  # save the new version away for next time...

        name = dialog.GetSimpleInput(
            "File name", lastLocateFileName, "Locate Python File"
        )
        if name is None:  # Cancelled.
            return
        lastLocateFileName = name
        # if ".py" supplied, rip it off!
        # should also check for .pys and .pyw
        if lastLocateFileName[-3:].lower() == ".py":
            lastLocateFileName = lastLocateFileName[:-3]
        lastLocateFileName = lastLocateFileName.replace(".", "\\")
        newName = scriptutils.LocatePythonFile(lastLocateFileName)
        if newName is None:
            win32ui.MessageBox("The file '%s' can not be located" % lastLocateFileName)
        else:
            win32ui.GetApp().OpenDocumentFile(newName)

    # Display all the "options" proprety pages we can find
    def OnViewOptions(self, id, code):
        win32ui.InitRichEdit()
        sheet = dialog.PropertySheet("Pythonwin Options")
        # Add property pages we know about that need manual work.
        from pywin.dialogs import ideoptions

        sheet.AddPage(ideoptions.OptionsPropPage())

        from . import toolmenu

        sheet.AddPage(toolmenu.ToolMenuPropPage())

        # Get other dynamic pages from templates.
        pages = []
        for template in self.GetDocTemplateList():
            try:
                # Dont actually call the function with the exception handler.
                getter = template.GetPythonPropertyPages
            except AttributeError:
                # Template does not provide property pages!
                continue
            pages = pages + getter()

        # Debugger template goes at the end
        try:
            from pywin.debugger import configui
        except ImportError:
            configui = None
        if configui is not None:
            pages.append(configui.DebuggerOptionsPropPage())
        # Now simply add the pages, and display the dialog.
        for page in pages:
            sheet.AddPage(page)

        if sheet.DoModal() == win32con.IDOK:
            win32ui.SetStatusText("Applying configuration changes...", 1)
            win32ui.DoWaitCursor(1)
            # Tell every Window in our app that win.ini has changed!
            win32ui.GetMainFrame().SendMessageToDescendants(
                win32con.WM_WININICHANGE, 0, 0
            )
            win32ui.DoWaitCursor(0)

    def OnInteractiveWindow(self, id, code):
        # toggle the existing state.
        from . import interact

        interact.ToggleInteractiveWindow()

    def OnUpdateInteractiveWindow(self, cmdui):
        try:
            interact = sys.modules["pywin.framework.interact"]
            state = interact.IsInteractiveWindowVisible()
        except KeyError:  # Interactive module hasnt ever been imported.
            state = 0
        cmdui.Enable()
        cmdui.SetCheck(state)

    def OnFileSaveAll(self, id, code):
        # Only attempt to save editor documents.
        from pywin.framework.editor import editorTemplate

        num = 0
        for doc in editorTemplate.GetDocumentList():
            if doc.IsModified() and doc.GetPathName():
                num = num = 1
                doc.OnSaveDocument(doc.GetPathName())
        win32ui.SetStatusText("%d documents saved" % num, 1)

    def OnViewToolbarDbg(self, id, code):
        if code == 0:
            return not win32ui.GetMainFrame().OnBarCheck(id)

    def OnUpdateViewToolbarDbg(self, cmdui):
        win32ui.GetMainFrame().OnUpdateControlBarMenu(cmdui)
        cmdui.Enable(1)

    def OnHelpIndex(self, id, code):
        from . import help

        help.SelectAndRunHelpFile()


# As per the comments in app.py, this use is depreciated.
# app.AppBuilder = InteractivePythonApp

# Now all we do is create the application
thisApp = InteractivePythonApp()
