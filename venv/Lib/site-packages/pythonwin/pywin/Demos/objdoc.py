# This is a sample file, and shows the basic framework for using an "Object" based
# document, rather than a "filename" based document.
# This is referenced by the Pythonwin .html documentation.

# In the example below, the OpenObject() method is used instead of OpenDocumentFile,
# and all the core MFC document open functionality is retained.

import win32ui
from pywin.mfc import docview


class object_template(docview.DocTemplate):
    def __init__(self):
        docview.DocTemplate.__init__(self, None, None, None, object_view)

    def OpenObject(self, object):  # Use this instead of OpenDocumentFile.
        # Look for existing open document
        for doc in self.GetDocumentList():
            print("document is ", doc)
            if doc.object is object:
                doc.GetFirstView().ActivateFrame()
                return doc
        # not found - new one.
        doc = object_document(self, object)
        frame = self.CreateNewFrame(doc)
        doc.OnNewDocument()
        doc.SetTitle(str(object))
        self.InitialUpdateFrame(frame, doc)
        return doc


class object_document(docview.Document):
    def __init__(self, template, object):
        docview.Document.__init__(self, template)
        self.object = object

    def OnOpenDocument(self, name):
        raise RuntimeError("Should not be called if template strings set up correctly")
        return 0


class object_view(docview.EditView):
    def OnInitialUpdate(self):
        self.ReplaceSel("Object is %s" % repr(self.GetDocument().object))


def demo():
    t = object_template()
    d = t.OpenObject(win32ui)
    return (t, d)


if __name__ == "__main__":
    import demoutils

    if demoutils.NeedGoodGUI():
        demo()
