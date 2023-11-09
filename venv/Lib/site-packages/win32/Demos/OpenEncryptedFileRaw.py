import os

import win32api
import win32file
import winerror


def ReadCallback(input_buffer, data, buflen):
    fnamein, fnameout, f = data
    ## print fnamein, fnameout, buflen
    f.write(input_buffer)
    ## python 2.3 throws an error if return value is a plain int
    return winerror.ERROR_SUCCESS


def WriteCallback(output_buffer, data, buflen):
    fnamebackup, fnameout, f = data
    file_data = f.read(buflen)
    ## returning 0 as len terminates WriteEncryptedFileRaw
    output_len = len(file_data)
    output_buffer[:output_len] = file_data
    return winerror.ERROR_SUCCESS, output_len


tmp_dir = win32api.GetTempPath()
dst_dir = win32api.GetTempFileName(tmp_dir, "oef")[0]
os.remove(dst_dir)
os.mkdir(dst_dir)
print("Destination dir:", dst_dir)

## create an encrypted file
fname = win32api.GetTempFileName(dst_dir, "ref")[0]
print("orig file:", fname)
f = open(fname, "w")
f.write("xxxxxxxxxxxxxxxx\n" * 32768)
f.close()
## add a couple of extra data streams
f = open(fname + ":stream_y", "w")
f.write("yyyyyyyyyyyyyyyy\n" * 32768)
f.close()
f = open(fname + ":stream_z", "w")
f.write("zzzzzzzzzzzzzzzz\n" * 32768)
f.close()
win32file.EncryptFile(fname)

## backup raw data of encrypted file
bkup_fname = win32api.GetTempFileName(dst_dir, "bef")[0]
print("backup file:", bkup_fname)
f = open(bkup_fname, "wb")
ctxt = win32file.OpenEncryptedFileRaw(fname, 0)
try:
    win32file.ReadEncryptedFileRaw(ReadCallback, (fname, bkup_fname, f), ctxt)
finally:
    ## if context is not closed, file remains locked even if calling process is killed
    win32file.CloseEncryptedFileRaw(ctxt)
    f.close()

## restore data from backup to new encrypted file
dst_fname = win32api.GetTempFileName(dst_dir, "wef")[0]
print("restored file:", dst_fname)
f = open(bkup_fname, "rb")
ctxtout = win32file.OpenEncryptedFileRaw(dst_fname, win32file.CREATE_FOR_IMPORT)
try:
    win32file.WriteEncryptedFileRaw(WriteCallback, (bkup_fname, dst_fname, f), ctxtout)
finally:
    win32file.CloseEncryptedFileRaw(ctxtout)
    f.close()
