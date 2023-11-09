## demonstrates using BackupSeek to enumerate data streams for a file
import struct

import pythoncom
import pywintypes
import win32api
import win32con
import win32file
from win32com import storagecon

stream_types = {
    win32con.BACKUP_DATA: "Standard data",
    win32con.BACKUP_EA_DATA: "Extended attribute data",
    win32con.BACKUP_SECURITY_DATA: "Security descriptor data",
    win32con.BACKUP_ALTERNATE_DATA: "Alternative data streams",
    win32con.BACKUP_LINK: "Hard link information",
    win32con.BACKUP_PROPERTY_DATA: "Property data",
    win32con.BACKUP_OBJECT_ID: "Objects identifiers",
    win32con.BACKUP_REPARSE_DATA: "Reparse points",
    win32con.BACKUP_SPARSE_BLOCK: "Sparse file",
}

tempdir = win32api.GetTempPath()
tempfile = win32api.GetTempFileName(tempdir, "bkr")[0]
print("Filename:", tempfile)

f = open(tempfile, "w")
f.write("some random junk" + "x" * 100)
f.close()

f = open(tempfile + ":streamdata", "w")
f.write("data written to alternate stream" + "y" * 100)
f.close()

f = open(tempfile + ":anotherstream", "w")
f.write("z" * 200)
f.close()

## add Summary Information, which is stored as a separate stream
m = storagecon.STGM_READWRITE | storagecon.STGM_SHARE_EXCLUSIVE | storagecon.STGM_DIRECT
pss = pythoncom.StgOpenStorageEx(
    tempfile, m, storagecon.STGFMT_FILE, 0, pythoncom.IID_IPropertySetStorage, None
)
ps = pss.Create(
    pythoncom.FMTID_SummaryInformation,
    pythoncom.IID_IPropertyStorage,
    0,
    storagecon.STGM_READWRITE | storagecon.STGM_SHARE_EXCLUSIVE,
)
ps.WriteMultiple(
    (storagecon.PIDSI_KEYWORDS, storagecon.PIDSI_COMMENTS), ("keywords", "comments")
)
ps = None
pss = None

sa = pywintypes.SECURITY_ATTRIBUTES()
sa.bInheritHandle = False
h = win32file.CreateFile(
    tempfile,
    win32con.GENERIC_ALL,
    win32con.FILE_SHARE_READ,
    sa,
    win32con.OPEN_EXISTING,
    win32file.FILE_FLAG_BACKUP_SEMANTICS,
    None,
)


""" stream header:
typedef struct _WIN32_STREAM_ID {
    DWORD dwStreamId;  DWORD dwStreamAttributes;  LARGE_INTEGER Size;
    DWORD dwStreamNameSize;  WCHAR cStreamName[ANYSIZE_ARRAY];
}
"""

win32_stream_id_format = "LLQL"
win32_stream_id_size = struct.calcsize(win32_stream_id_format)


def parse_stream_header(h, ctxt, data):
    stream_type, stream_attributes, stream_size, stream_name_size = struct.unpack(
        win32_stream_id_format, data
    )
    print(
        "\nType:",
        stream_type,
        stream_types[stream_type],
        "Attributes:",
        stream_attributes,
        "Size:",
        stream_size,
        "Name len:",
        stream_name_size,
    )
    if stream_name_size > 0:
        ## ??? sdk says this size is in characters, but it appears to be number of bytes ???
        bytes_read, stream_name_buf, ctxt = win32file.BackupRead(
            h, stream_name_size, None, False, True, ctxt
        )
        stream_name = pywintypes.UnicodeFromRaw(stream_name_buf[:])
    else:
        stream_name = "Unnamed"
    print("Name:" + stream_name)
    return (
        ctxt,
        stream_type,
        stream_attributes,
        stream_size,
        stream_name_size,
        stream_name,
    )


ctxt = 0
win32_stream_id_buf = (
    None  ## gets rebound to a writable buffer on first call and reused
)
while 1:
    bytes_read, win32_stream_id_buf, ctxt = win32file.BackupRead(
        h, win32_stream_id_size, win32_stream_id_buf, False, True, ctxt
    )
    if bytes_read == 0:
        break
    (
        ctxt,
        stream_type,
        stream_attributes,
        stream_size,
        stream_name_size,
        stream_name,
    ) = parse_stream_header(h, ctxt, win32_stream_id_buf[:])
    if stream_size > 0:
        bytes_moved = win32file.BackupSeek(h, stream_size, ctxt)
        print("Moved: ", bytes_moved)

win32file.BackupRead(h, win32_stream_id_size, win32_stream_id_buf, True, True, ctxt)
win32file.CloseHandle(h)
