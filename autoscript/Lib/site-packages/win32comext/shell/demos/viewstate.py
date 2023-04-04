"""
Demonstrates how to propagate a folder's view state to all its subfolders
The format of the ColInfo stream is apparently undocumented, but
it can be read raw from one folder and copied to another's view state.
"""

import os
import sys

import pythoncom
from win32com.shell import shell, shellcon

template_folder = os.path.split(sys.executable)[0]
print("Template folder:", template_folder)
template_pidl = shell.SHILCreateFromPath(template_folder, 0)[0]
template_pb = shell.SHGetViewStatePropertyBag(
    template_pidl,
    "Shell",
    shellcon.SHGVSPB_FOLDERNODEFAULTS,
    pythoncom.IID_IPropertyBag,
)

# Column info has to be read as a stream
# This may blow up if folder has never been opened in Explorer and has no ColInfo yet
template_iunk = template_pb.Read("ColInfo", pythoncom.VT_UNKNOWN)
template_stream = template_iunk.QueryInterface(pythoncom.IID_IStream)
streamsize = template_stream.Stat()[2]
template_colinfo = template_stream.Read(streamsize)


def update_colinfo(not_used, dir_name, fnames):
    for fname in fnames:
        full_fname = os.path.join(dir_name, fname)
        if os.path.isdir(full_fname):
            print(full_fname)
            pidl = shell.SHILCreateFromPath(full_fname, 0)[0]
            pb = shell.SHGetViewStatePropertyBag(
                pidl,
                "Shell",
                shellcon.SHGVSPB_FOLDERNODEFAULTS,
                pythoncom.IID_IPropertyBag,
            )
            ## not all folders already have column info, and we're replacing it anyway
            pb.Write("ColInfo", template_stream)
            iunk = pb.Read("ColInfo", pythoncom.VT_UNKNOWN)
            s = iunk.QueryInterface(pythoncom.IID_IStream)
            s.Write(template_colinfo)
            s = None
            ## attribute names read from registry, can't find any way to enumerate IPropertyBag
            for attr in (
                "Address",
                "Buttons",
                "Col",
                "Vid",
                "WFlags",
                "FFlags",
                "Sort",
                "SortDir",
                "ShowCmd",
                "FolderType",
                "Mode",
                "Rev",
            ):
                pb.Write(attr, template_pb.Read(attr))
            pb = None


os.path.walk(template_folder, update_colinfo, None)
