import traceback

import pythoncom
from win32com.axdebug import axdebug
from win32com.client.util import Enumerator


def DumpDebugApplicationNode(node, level=0):
    # Recursive dump of a DebugApplicationNode
    spacer = " " * level
    for desc, attr in [
        ("Node Name", axdebug.DOCUMENTNAMETYPE_APPNODE),
        ("Title", axdebug.DOCUMENTNAMETYPE_TITLE),
        ("Filename", axdebug.DOCUMENTNAMETYPE_FILE_TAIL),
        ("URL", axdebug.DOCUMENTNAMETYPE_URL),
    ]:
        try:
            info = node.GetName(attr)
        except pythoncom.com_error:
            info = "<N/A>"
        print("%s%s: %s" % (spacer, desc, info))
    try:
        doc = node.GetDocument()
    except pythoncom.com_error:
        doc = None
    if doc:
        doctext = doc.QueryInterface(axdebug.IID_IDebugDocumentText)
        numLines, numChars = doctext.GetSize()
        #                       text, attr = doctext.GetText(0, 20, 1)
        text, attr = doctext.GetText(0, numChars, 1)
        print(
            "%sText is %s, %d bytes long" % (spacer, repr(text[:40] + "..."), len(text))
        )
    else:
        print("%s%s" % (spacer, "<No document available>"))

    for child in Enumerator(node.EnumChildren()):
        DumpDebugApplicationNode(child, level + 1)


def dumpall():
    dm = pythoncom.CoCreateInstance(
        axdebug.CLSID_MachineDebugManager,
        None,
        pythoncom.CLSCTX_ALL,
        axdebug.IID_IMachineDebugManager,
    )
    e = Enumerator(dm.EnumApplications())
    for app in e:
        print("Application: %s" % app.GetName())
        node = (
            app.GetRootNode()
        )  # of type PyIDebugApplicationNode->PyIDebugDocumentProvider->PyIDebugDocumentInfo
        DumpDebugApplicationNode(node)


if __name__ == "__main__":
    try:
        dumpall()
    except:
        traceback.print_exc()
