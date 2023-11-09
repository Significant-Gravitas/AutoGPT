import os

import pywin.framework.window
import win32api
import win32ui
from pywin.mfc import docview

from . import frame

ParentEditorTemplate = docview.DocTemplate


class EditorTemplateBase(ParentEditorTemplate):
    def __init__(
        self, res=win32ui.IDR_TEXTTYPE, makeDoc=None, makeFrame=None, makeView=None
    ):
        if makeFrame is None:
            makeFrame = frame.EditorFrame
        ParentEditorTemplate.__init__(self, res, makeDoc, makeFrame, makeView)

    def _CreateDocTemplate(self, resourceId):
        assert 0, "You must override this"

    def CreateWin32uiDocument(self):
        assert 0, "You must override this"

    def GetFileExtensions(self):
        return ".txt", ".py"

    def MatchDocType(self, fileName, fileType):
        doc = self.FindOpenDocument(fileName)
        if doc:
            return doc
        ext = os.path.splitext(fileName)[1].lower()
        if ext in self.GetFileExtensions():
            return win32ui.CDocTemplate_Confidence_yesAttemptNative
        return win32ui.CDocTemplate_Confidence_maybeAttemptForeign

    def InitialUpdateFrame(self, frame, doc, makeVisible=1):
        self._obj_.InitialUpdateFrame(frame, doc, makeVisible)  # call default handler.
        doc._UpdateUIForState()

    def GetPythonPropertyPages(self):
        """Returns a list of property pages"""
        from . import configui

        return [configui.EditorPropertyPage(), configui.EditorWhitespacePropertyPage()]

    def OpenDocumentFile(self, filename, bMakeVisible=1):
        if filename is not None:
            try:
                path = os.path.split(filename)[0]
                # 				print "The editor is translating", `filename`,"to",
                filename = win32api.FindFiles(filename)[0][8]
                filename = os.path.join(path, filename)
            # 				print `filename`
            except (win32api.error, IndexError) as details:
                pass
        # 				print "Couldnt get the full filename!", details
        return self._obj_.OpenDocumentFile(filename, bMakeVisible)
