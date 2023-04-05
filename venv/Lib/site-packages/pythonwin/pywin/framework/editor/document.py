# We no longer support the old, non-colour editor!

import os
import shutil
import traceback

import win32api
import win32con
import win32ui
from pywin.framework.editor import GetEditorOption
from pywin.mfc import docview, object

BAK_NONE = 0
BAK_DOT_BAK = 1
BAK_DOT_BAK_TEMP_DIR = 2
BAK_DOT_BAK_BAK_DIR = 3

MSG_CHECK_EXTERNAL_FILE = (
    win32con.WM_USER + 1999
)  ## WARNING: Duplicated in editor.py and coloreditor.py

import pywin.scintilla.document

ParentEditorDocument = pywin.scintilla.document.CScintillaDocument


class EditorDocumentBase(ParentEditorDocument):
    def __init__(self, template):
        self.bAutoReload = GetEditorOption("Auto Reload", 1)
        self.bDeclinedReload = 0  # Has the user declined to reload.
        self.fileStat = None
        self.bReportedFileNotFound = 0

        # what sort of bak file should I create.
        # default to write to %temp%/bak/filename.ext
        self.bakFileType = GetEditorOption("Backup Type", BAK_DOT_BAK_BAK_DIR)

        self.watcherThread = FileWatchingThread(self)
        self.watcherThread.CreateThread()
        # Should I try and use VSS integration?
        self.scModuleName = GetEditorOption("Source Control Module", "")
        self.scModule = None  # Loaded when first used.
        ParentEditorDocument.__init__(self, template, template.CreateWin32uiDocument())

    def OnCloseDocument(self):
        self.watcherThread.SignalStop()
        return self._obj_.OnCloseDocument()

    # 	def OnOpenDocument(self, name):
    # 		rc = ParentEditorDocument.OnOpenDocument(self, name)
    # 		self.GetFirstView()._SetLoadedText(self.text)
    # 		self._DocumentStateChanged()
    # 		return rc

    def OnSaveDocument(self, fileName):
        win32ui.SetStatusText("Saving file...", 1)
        # rename to bak if required.
        dir, basename = os.path.split(fileName)
        if self.bakFileType == BAK_DOT_BAK:
            bakFileName = dir + "\\" + os.path.splitext(basename)[0] + ".bak"
        elif self.bakFileType == BAK_DOT_BAK_TEMP_DIR:
            bakFileName = (
                win32api.GetTempPath() + "\\" + os.path.splitext(basename)[0] + ".bak"
            )
        elif self.bakFileType == BAK_DOT_BAK_BAK_DIR:
            tempPath = os.path.join(win32api.GetTempPath(), "bak")
            try:
                os.mkdir(tempPath, 0)
            except os.error:
                pass
            bakFileName = os.path.join(tempPath, basename)
        try:
            os.unlink(bakFileName)  # raise NameError if no bakups wanted.
        except (os.error, NameError):
            pass
        try:
            # Do a copy as it might be on different volumes,
            # and the file may be a hard-link, causing the link
            # to follow the backup.
            shutil.copy2(fileName, bakFileName)
        except (os.error, NameError, IOError):
            pass
        try:
            self.SaveFile(fileName)
        except IOError as details:
            win32ui.MessageBox("Error - could not save file\r\n\r\n%s" % details)
            return 0
        except (UnicodeEncodeError, LookupError) as details:
            rc = win32ui.MessageBox(
                "Encoding failed: \r\n%s" % details
                + "\r\nPlease add desired source encoding as first line of file, eg \r\n"
                + "# -*- coding: mbcs -*-\r\n\r\n"
                + "If you continue, the file will be saved as binary and will\r\n"
                + "not be valid in the declared encoding.\r\n\r\n"
                + "Save the file as binary with an invalid encoding?",
                "File save failed",
                win32con.MB_YESNO | win32con.MB_DEFBUTTON2,
            )
            if rc == win32con.IDYES:
                try:
                    self.SaveFile(fileName, encoding="latin-1")
                except IOError as details:
                    win32ui.MessageBox(
                        "Error - could not save file\r\n\r\n%s" % details
                    )
                    return 0
            else:
                return 0
        self.SetModifiedFlag(0)  # No longer dirty
        self.bDeclinedReload = 0  # They probably want to know if it changes again!
        win32ui.AddToRecentFileList(fileName)
        self.SetPathName(fileName)
        win32ui.SetStatusText("Ready")
        self._DocumentStateChanged()
        return 1

    def FinalizeViewCreation(self, view):
        ParentEditorDocument.FinalizeViewCreation(self, view)
        if view == self.GetFirstView():
            self._DocumentStateChanged()
            if view.bFolding and GetEditorOption("Fold On Open", 0):
                view.FoldTopLevelEvent()

    def HookViewNotifications(self, view):
        ParentEditorDocument.HookViewNotifications(self, view)

    # Support for reloading the document from disk - presumably after some
    # external application has modified it (or possibly source control has
    # checked it out.
    def ReloadDocument(self):
        """Reloads the document from disk.  Assumes the file has
        been saved and user has been asked if necessary - it just does it!
        """
        win32ui.SetStatusText("Reloading document.  Please wait...", 1)
        self.SetModifiedFlag(0)
        # Loop over all views, saving their state, then reload the document
        views = self.GetAllViews()
        states = []
        for view in views:
            try:
                info = view._PrepareUserStateChange()
            except AttributeError:  # Not our editor view?
                info = None
            states.append(info)
        self.OnOpenDocument(self.GetPathName())
        for view, info in zip(views, states):
            if info is not None:
                view._EndUserStateChange(info)
        self._DocumentStateChanged()
        win32ui.SetStatusText("Document reloaded.")

    # Reloading the file
    def CheckExternalDocumentUpdated(self):
        if self.bDeclinedReload or not self.GetPathName():
            return
        try:
            newstat = os.stat(self.GetPathName())
        except os.error as exc:
            if not self.bReportedFileNotFound:
                print(
                    "The file '%s' is open for editing, but\nchecking it for changes caused the error: %s"
                    % (self.GetPathName(), exc.strerror)
                )
                self.bReportedFileNotFound = 1
            return
        if self.bReportedFileNotFound:
            print(
                "The file '%s' has re-appeared - continuing to watch for changes..."
                % (self.GetPathName(),)
            )
            self.bReportedFileNotFound = (
                0  # Once found again we want to start complaining.
            )
        changed = (
            (self.fileStat is None)
            or self.fileStat[0] != newstat[0]
            or self.fileStat[6] != newstat[6]
            or self.fileStat[8] != newstat[8]
            or self.fileStat[9] != newstat[9]
        )
        if changed:
            question = None
            if self.IsModified():
                question = (
                    "%s\r\n\r\nThis file has been modified outside of the source editor.\r\nDo you want to reload it and LOSE THE CHANGES in the source editor?"
                    % self.GetPathName()
                )
                mbStyle = win32con.MB_YESNO | win32con.MB_DEFBUTTON2  # Default to "No"
            else:
                if not self.bAutoReload:
                    question = (
                        "%s\r\n\r\nThis file has been modified outside of the source editor.\r\nDo you want to reload it?"
                        % self.GetPathName()
                    )
                    mbStyle = win32con.MB_YESNO  # Default to "Yes"
            if question:
                rc = win32ui.MessageBox(question, None, mbStyle)
                if rc != win32con.IDYES:
                    self.bDeclinedReload = 1
                    return
            self.ReloadDocument()

    def _DocumentStateChanged(self):
        """Called whenever the documents state (on disk etc) has been changed
        by the editor (eg, as the result of a save operation)
        """
        if self.GetPathName():
            try:
                self.fileStat = os.stat(self.GetPathName())
            except os.error:
                self.fileStat = None
        else:
            self.fileStat = None
        self.watcherThread._DocumentStateChanged()
        self._UpdateUIForState()
        self._ApplyOptionalToViews("_UpdateUIForState")
        self._ApplyOptionalToViews("SetReadOnly", self._IsReadOnly())
        self._ApplyOptionalToViews("SCISetSavePoint")
        # Allow the debugger to reset us too.
        import pywin.debugger

        if pywin.debugger.currentDebugger is not None:
            pywin.debugger.currentDebugger.UpdateDocumentLineStates(self)

    # Read-only document support - make it obvious to the user
    # that the file is read-only.
    def _IsReadOnly(self):
        return self.fileStat is not None and (self.fileStat[0] & 128) == 0

    def _UpdateUIForState(self):
        """Change the title to reflect the state of the document -
        eg ReadOnly, Dirty, etc
        """
        filename = self.GetPathName()
        if not filename:
            return  # New file - nothing to do
        try:
            # This seems necessary so the internal state of the window becomes
            # "visible".  without it, it is still shown, but certain functions
            # (such as updating the title) dont immediately work?
            self.GetFirstView().ShowWindow(win32con.SW_SHOW)
            title = win32ui.GetFileTitle(filename)
        except win32ui.error:
            title = filename
        if self._IsReadOnly():
            title = title + " (read-only)"
        self.SetTitle(title)

    def MakeDocumentWritable(self):
        pretend_ss = 0  # Set to 1 to test this without source safe :-)
        if not self.scModuleName and not pretend_ss:  # No Source Control support.
            win32ui.SetStatusText(
                "Document is read-only, and no source-control system is configured"
            )
            win32api.MessageBeep()
            return 0

        # We have source control support - check if the user wants to use it.
        msg = "Would you like to check this file out?"
        defButton = win32con.MB_YESNO
        if self.IsModified():
            msg = msg + "\r\n\r\nALL CHANGES IN THE EDITOR WILL BE LOST"
            defButton = win32con.MB_YESNO
        if win32ui.MessageBox(msg, None, defButton) != win32con.IDYES:
            return 0

        if pretend_ss:
            print("We are only pretending to check it out!")
            win32api.SetFileAttributes(
                self.GetPathName(), win32con.FILE_ATTRIBUTE_NORMAL
            )
            self.ReloadDocument()
            return 1

        # Now call on the module to do it.
        if self.scModule is None:
            try:
                self.scModule = __import__(self.scModuleName)
                for part in self.scModuleName.split(".")[1:]:
                    self.scModule = getattr(self.scModule, part)
            except:
                traceback.print_exc()
                print("Error loading source control module.")
                return 0

        if self.scModule.CheckoutFile(self.GetPathName()):
            self.ReloadDocument()
            return 1
        return 0

    def CheckMakeDocumentWritable(self):
        if self._IsReadOnly():
            return self.MakeDocumentWritable()
        return 1

    def SaveModified(self):
        # Called as the document is closed.  If we are about
        # to prompt for a save, bring the document to the foreground.
        if self.IsModified():
            frame = self.GetFirstView().GetParentFrame()
            try:
                frame.MDIActivate()
                frame.AutoRestore()
            except:
                print("Could not bring document to foreground")
        return self._obj_.SaveModified()


# NOTE - I DONT use the standard threading module,
# as this waits for all threads to terminate at shutdown.
# When using the debugger, it is possible shutdown will
# occur without Pythonwin getting a complete shutdown,
# so we deadlock at the end - threading is waiting for
import pywin.mfc.thread
import win32event


class FileWatchingThread(pywin.mfc.thread.WinThread):
    def __init__(self, doc):
        self.doc = doc
        self.adminEvent = win32event.CreateEvent(None, 0, 0, None)
        self.stopEvent = win32event.CreateEvent(None, 0, 0, None)
        self.watchEvent = None
        pywin.mfc.thread.WinThread.__init__(self)

    def _DocumentStateChanged(self):
        win32event.SetEvent(self.adminEvent)

    def RefreshEvent(self):
        self.hwnd = self.doc.GetFirstView().GetSafeHwnd()
        if self.watchEvent is not None:
            win32api.FindCloseChangeNotification(self.watchEvent)
            self.watchEvent = None
        path = self.doc.GetPathName()
        if path:
            path = os.path.dirname(path)
        if path:
            filter = (
                win32con.FILE_NOTIFY_CHANGE_FILE_NAME
                | win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES
                | win32con.FILE_NOTIFY_CHANGE_LAST_WRITE
            )
            try:
                self.watchEvent = win32api.FindFirstChangeNotification(path, 0, filter)
            except win32api.error as exc:
                print("Can not watch file", path, "for changes -", exc.strerror)

    def SignalStop(self):
        win32event.SetEvent(self.stopEvent)

    def Run(self):
        while 1:
            handles = [self.stopEvent, self.adminEvent]
            if self.watchEvent is not None:
                handles.append(self.watchEvent)
            rc = win32event.WaitForMultipleObjects(handles, 0, win32event.INFINITE)
            if rc == win32event.WAIT_OBJECT_0:
                break
            elif rc == win32event.WAIT_OBJECT_0 + 1:
                self.RefreshEvent()
            else:
                win32api.PostMessage(self.hwnd, MSG_CHECK_EXTERNAL_FILE, 0, 0)
                try:
                    # If the directory has been removed underneath us, we get this error.
                    win32api.FindNextChangeNotification(self.watchEvent)
                except win32api.error as exc:
                    print(
                        "Can not watch file",
                        self.doc.GetPathName(),
                        "for changes -",
                        exc.strerror,
                    )
                    break

        # close a circular reference
        self.doc = None
        if self.watchEvent:
            win32api.FindCloseChangeNotification(self.watchEvent)
