## demonstrates using BackupRead and BackupWrite to copy all of a file's data streams


import ntsecuritycon
import pythoncom
import pywintypes
import win32api
import win32con
import win32file
import win32security
from pywin32_testutil import ob2memory, str2bytes
from win32com import storagecon

all_sd_info = (
    win32security.DACL_SECURITY_INFORMATION
    | win32security.DACL_SECURITY_INFORMATION
    | win32security.OWNER_SECURITY_INFORMATION
    | win32security.GROUP_SECURITY_INFORMATION
)

tempdir = win32api.GetTempPath()
tempfile = win32api.GetTempFileName(tempdir, "bkr")[0]
outfile = win32api.GetTempFileName(tempdir, "out")[0]
print("Filename:", tempfile, "Output file:", outfile)

f = open(tempfile, "w")
f.write("some random junk" + "x" * 100)
f.close()

## add a couple of alternate data streams
f = open(tempfile + ":streamdata", "w")
f.write("data written to alternate stream" + "y" * 100)
f.close()

f = open(tempfile + ":anotherstream", "w")
f.write("z" * 100)
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

## add a custom security descriptor to make sure we don't
##   get a default that would always be the same for both files in temp dir
new_sd = pywintypes.SECURITY_DESCRIPTOR()
sid = win32security.LookupAccountName("", "EveryOne")[0]
acl = pywintypes.ACL()
acl.AddAccessAllowedAce(1, win32con.GENERIC_READ, sid)
acl.AddAccessAllowedAce(1, ntsecuritycon.FILE_APPEND_DATA, sid)
acl.AddAccessAllowedAce(1, win32con.GENERIC_WRITE, sid)
acl.AddAccessAllowedAce(1, ntsecuritycon.FILE_ALL_ACCESS, sid)

new_sd.SetSecurityDescriptorDacl(True, acl, False)
win32security.SetFileSecurity(tempfile, win32security.DACL_SECURITY_INFORMATION, new_sd)


sa = pywintypes.SECURITY_ATTRIBUTES()
sa.bInheritHandle = True
h = win32file.CreateFile(
    tempfile,
    win32con.GENERIC_ALL,
    win32con.FILE_SHARE_READ,
    sa,
    win32con.OPEN_EXISTING,
    win32file.FILE_FLAG_BACKUP_SEMANTICS,
    None,
)

outh = win32file.CreateFile(
    outfile,
    win32con.GENERIC_ALL,
    win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
    sa,
    win32con.OPEN_EXISTING,
    win32file.FILE_FLAG_BACKUP_SEMANTICS,
    None,
)

ctxt = 0
outctxt = 0
buf = None
readsize = 100

while 1:
    bytes_read, buf, ctxt = win32file.BackupRead(h, readsize, buf, False, True, ctxt)
    if bytes_read == 0:
        break
    bytes_written, outctxt = win32file.BackupWrite(
        outh, bytes_read, buf, False, True, outctxt
    )
    print("Written:", bytes_written, "Context:", outctxt)
win32file.BackupRead(h, 0, buf, True, True, ctxt)
win32file.BackupWrite(outh, 0, str2bytes(""), True, True, outctxt)
win32file.CloseHandle(h)
win32file.CloseHandle(outh)

assert open(tempfile).read() == open(outfile).read(), "File contents differ !"
assert (
    open(tempfile + ":streamdata").read() == open(outfile + ":streamdata").read()
), "streamdata contents differ !"
assert (
    open(tempfile + ":anotherstream").read() == open(outfile + ":anotherstream").read()
), "anotherstream contents differ !"
assert (
    ob2memory(win32security.GetFileSecurity(tempfile, all_sd_info))[:]
    == ob2memory(win32security.GetFileSecurity(outfile, all_sd_info))[:]
), "Security descriptors are different !"
## also should check Summary Info programatically
