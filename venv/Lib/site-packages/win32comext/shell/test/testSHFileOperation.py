import os

import win32api
from win32com.shell import shell, shellcon


def testSHFileOperation(file_cnt):
    temp_dir = os.environ["temp"]
    orig_fnames = [
        win32api.GetTempFileName(temp_dir, "sfo")[0] for x in range(file_cnt)
    ]
    new_fnames = [
        os.path.join(temp_dir, "copy of " + os.path.split(orig_fnames[x])[1])
        for x in range(file_cnt)
    ]

    pFrom = "\0".join(orig_fnames)
    pTo = "\0".join(new_fnames)

    shell.SHFileOperation(
        (
            0,
            shellcon.FO_MOVE,
            pFrom,
            pTo,
            shellcon.FOF_MULTIDESTFILES | shellcon.FOF_NOCONFIRMATION,
        )
    )
    for fname in orig_fnames:
        assert not os.path.isfile(fname)

    for fname in new_fnames:
        assert os.path.isfile(fname)
        shell.SHFileOperation(
            (
                0,
                shellcon.FO_DELETE,
                fname,
                None,
                shellcon.FOF_NOCONFIRMATION | shellcon.FOF_NOERRORUI,
            )
        )


def testSHNAMEMAPPINGS(file_cnt):
    ## attemps to move a set of files to names that already exist, and generated filenames should be returned
    ##   as a sequence of 2-tuples created from SHNAMEMAPPINGS handle
    temp_dir = os.environ["temp"]
    orig_fnames = [
        win32api.GetTempFileName(temp_dir, "sfo")[0] for x in range(file_cnt)
    ]
    new_fnames = [win32api.GetTempFileName(temp_dir, "sfo")[0] for x in range(file_cnt)]
    pFrom = "\0".join(orig_fnames)
    pTo = "\0".join(new_fnames)
    rc, banyaborted, NameMappings = shell.SHFileOperation(
        (
            0,
            shellcon.FO_MOVE,
            pFrom,
            pTo,
            shellcon.FOF_MULTIDESTFILES
            | shellcon.FOF_NOCONFIRMATION
            | shellcon.FOF_RENAMEONCOLLISION
            | shellcon.FOF_WANTMAPPINGHANDLE,
        )
    )

    for old_fname, new_fname in NameMappings:
        print("Old:", old_fname, "New:", new_fname)
    assert len(NameMappings) == file_cnt


testSHFileOperation(10)
testSHFileOperation(1)
testSHNAMEMAPPINGS(5)
