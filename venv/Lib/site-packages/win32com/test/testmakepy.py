# Test makepy - try and run it over every OCX in the windows system directory.

import sys
import traceback

import pythoncom
import win32api
import win32com.test.util
import winerror
from win32com.client import gencache, makepy, selecttlb


def TestBuildAll(verbose=1):
    num = 0
    tlbInfos = selecttlb.EnumTlbs()
    for info in tlbInfos:
        if verbose:
            print("%s (%s)" % (info.desc, info.dll))
        try:
            makepy.GenerateFromTypeLibSpec(info)
            #          sys.stderr.write("Attr typeflags for coclass referenced object %s=%d (%d), typekind=%d\n" % (name, refAttr.wTypeFlags, refAttr.wTypeFlags & pythoncom.TYPEFLAG_FDUAL,refAttr.typekind))
            num += 1
        except pythoncom.com_error as details:
            # Ignore these 2 errors, as the are very common and can obscure
            # useful warnings.
            if details.hresult not in [
                winerror.TYPE_E_CANTLOADLIBRARY,
                winerror.TYPE_E_LIBNOTREGISTERED,
            ]:
                print("** COM error on", info.desc)
                print(details)
        except KeyboardInterrupt:
            print("Interrupted!")
            raise KeyboardInterrupt
        except:
            print("Failed:", info.desc)
            traceback.print_exc()
        if makepy.bForDemandDefault:
            # This only builds enums etc by default - build each
            # interface manually
            tinfo = (info.clsid, info.lcid, info.major, info.minor)
            mod = gencache.EnsureModule(info.clsid, info.lcid, info.major, info.minor)
            for name in mod.NamesToIIDMap.keys():
                makepy.GenerateChildFromTypeLibSpec(name, tinfo)
    return num


def TestAll(verbose=0):
    num = TestBuildAll(verbose)
    print("Generated and imported", num, "modules")
    win32com.test.util.CheckClean()


if __name__ == "__main__":
    TestAll("-q" not in sys.argv)
