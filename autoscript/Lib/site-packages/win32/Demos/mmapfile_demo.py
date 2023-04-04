import os
import tempfile

import mmapfile
import win32api
import winerror
from pywin32_testutil import str2bytes

system_info = win32api.GetSystemInfo()
page_size = system_info[1]
alloc_size = system_info[7]

fname = tempfile.mktemp()
mapping_name = os.path.split(fname)[1]
fsize = 8 * page_size
print(fname, fsize, mapping_name)

m1 = mmapfile.mmapfile(File=fname, Name=mapping_name, MaximumSize=fsize)
m1.seek(100)
m1.write_byte(str2bytes("?"))
m1.seek(-1, 1)
assert m1.read_byte() == str2bytes("?")

## A reopened named mapping should have exact same size as original mapping
m2 = mmapfile.mmapfile(Name=mapping_name, File=None, MaximumSize=fsize * 2)
assert m2.size() == m1.size()
m1.seek(0, 0)
m1.write(fsize * str2bytes("s"))
assert m2.read(fsize) == fsize * str2bytes("s")

move_src = 100
move_dest = 500
move_size = 150

m2.seek(move_src, 0)
assert m2.tell() == move_src
m2.write(str2bytes("m") * move_size)
m2.move(move_dest, move_src, move_size)
m2.seek(move_dest, 0)
assert m2.read(move_size) == str2bytes("m") * move_size
##    m2.write('x'* (fsize+1))

m2.close()
m1.resize(fsize * 2)
assert m1.size() == fsize * 2
m1.seek(fsize)
m1.write(str2bytes("w") * fsize)
m1.flush()
m1.close()
os.remove(fname)


## Test a file with size larger than 32 bits
## need 10 GB free on drive where your temp folder lives
fname_large = tempfile.mktemp()
mapping_name = "Pywin32_large_mmap"
offsetdata = str2bytes("This is start of offset")

## Deliberately use odd numbers to test rounding logic
fsize = (1024 * 1024 * 1024 * 10) + 333
offset = (1024 * 1024 * 32) + 42
view_size = (1024 * 1024 * 16) + 111

## round mapping size and view size up to multiple of system page size
if fsize % page_size:
    fsize += page_size - (fsize % page_size)
if view_size % page_size:
    view_size += page_size - (view_size % page_size)
## round offset down to multiple of allocation granularity
offset -= offset % alloc_size

m1 = None
m2 = None
try:
    try:
        m1 = mmapfile.mmapfile(fname_large, mapping_name, fsize, 0, offset * 2)
    except mmapfile.error as exc:
        # if we don't have enough disk-space, that's OK.
        if exc.winerror != winerror.ERROR_DISK_FULL:
            raise
        print("skipping large file test - need", fsize, "available bytes.")
    else:
        m1.seek(offset)
        m1.write(offsetdata)

        ## When reopening an existing mapping without passing a file handle, you have
        ##  to specify a positive size even though it's ignored
        m2 = mmapfile.mmapfile(
            File=None,
            Name=mapping_name,
            MaximumSize=1,
            FileOffset=offset,
            NumberOfBytesToMap=view_size,
        )
        assert m2.read(len(offsetdata)) == offsetdata
finally:
    if m1 is not None:
        m1.close()
    if m2 is not None:
        m2.close()
    if os.path.exists(fname_large):
        os.remove(fname_large)
