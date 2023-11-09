# frame.py - The MDI frame window for an editor.
import pywin.framework.window
import win32con
import win32ui

from . import ModuleBrowser


class EditorFrame(pywin.framework.window.MDIChildWnd):
    def OnCreateClient(self, cp, context):
        # Create the default view as specified by the template (ie, the editor view)
        view = context.template.MakeView(context.doc)
        # Create the browser view.
        browserView = ModuleBrowser.BrowserView(context.doc)
        view2 = context.template.MakeView(context.doc)

        splitter = win32ui.CreateSplitter()
        style = win32con.WS_CHILD | win32con.WS_VISIBLE
        splitter.CreateStatic(self, 1, 2, style, win32ui.AFX_IDW_PANE_FIRST)
        sub_splitter = self.sub_splitter = win32ui.CreateSplitter()
        sub_splitter.CreateStatic(splitter, 2, 1, style, win32ui.AFX_IDW_PANE_FIRST + 1)

        # Note we must add the default view first, so that doc.GetFirstView() returns the editor view.
        sub_splitter.CreateView(view, 1, 0, (0, 0))
        splitter.CreateView(browserView, 0, 0, (0, 0))
        sub_splitter.CreateView(view2, 0, 0, (0, 0))

        ##        print "First view is", context.doc.GetFirstView()
        ##        print "Views are", view, view2, browserView
        ##        print "Parents are", view.GetParent(), view2.GetParent(), browserView.GetParent()
        ##        print "Splitter is", splitter
        ##        print "sub splitter is", sub_splitter
        ## Old
        ##        splitter.CreateStatic (self, 1, 2)
        ##        splitter.CreateView(view, 0, 1, (0,0)) # size ignored.
        ##        splitter.CreateView (browserView, 0, 0, (0, 0))

        # Restrict the size of the browser splitter (and we can avoid filling
        # it until it is shown)
        splitter.SetColumnInfo(0, 10, 20)
        # And the active view is our default view (so it gets initial focus)
        self.SetActiveView(view)

    def GetEditorView(self):
        # In a multi-view (eg, splitter) environment, get
        # an editor (ie, scintilla) view
        # Look for the splitter opened the most!
        if self.sub_splitter is None:
            return self.GetDlgItem(win32ui.AFX_IDW_PANE_FIRST)
        v1 = self.sub_splitter.GetPane(0, 0)
        v2 = self.sub_splitter.GetPane(1, 0)
        r1 = v1.GetWindowRect()
        r2 = v2.GetWindowRect()
        if r1[3] - r1[1] > r2[3] - r2[1]:
            return v1
        return v2

    def GetBrowserView(self):
        # XXX - should fix this :-)
        return self.GetActiveDocument().GetAllViews()[1]

    def OnClose(self):
        doc = self.GetActiveDocument()
        if not doc.SaveModified():
            ## Cancel button selected from Save dialog, do not actually close
            ## print 'close cancelled'
            return 0
        ## So the 'Save' dialog doesn't come up twice
        doc._obj_.SetModifiedFlag(False)

        # Must force the module browser to close itself here (OnDestroy for the view itself is too late!)
        self.sub_splitter = None  # ensure no circles!
        self.GetBrowserView().DestroyBrowser()
        return self._obj_.OnClose()
