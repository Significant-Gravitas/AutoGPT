# Originally written by Curt Hagenlocher, and various bits
# and pieces by Mark Hammond (and now Greg Stein has had
# a go too :-)

# Note that the main worker code has been moved to genpy.py
# As this is normally run from the command line, it reparses the code each time.
# Now this is nothing more than the command line handler and public interface.

# XXX - TO DO
# XXX - Greg and Mark have some ideas for a revamp - just no
#       time - if you want to help, contact us for details.
#       Main idea is to drop the classes exported and move to a more
#       traditional data driven model.

"""Generate a .py file from an OLE TypeLibrary file.


 This module is concerned only with the actual writing of
 a .py file.  It draws on the @build@ module, which builds
 the knowledge of a COM interface.

"""
usageHelp = """ \

Usage:

  makepy.py [-i] [-v|q] [-h] [-u] [-o output_file] [-d] [typelib, ...]

  -i    -- Show information for the specified typelib.

  -v    -- Verbose output.

  -q    -- Quiet output.

  -h    -- Do not generate hidden methods.

  -u    -- Python 1.5 and earlier: Do NOT convert all Unicode objects to
           strings.

           Python 1.6 and later: Convert all Unicode objects to strings.

  -o    -- Create output in a specified output file.  If the path leading
           to the file does not exist, any missing directories will be
           created.
           NOTE: -o cannot be used with -d.  This will generate an error.

  -d    -- Generate the base code now and the class code on demand.
           Recommended for large type libraries.

  typelib -- A TLB, DLL, OCX or anything containing COM type information.
             If a typelib is not specified, a window containing a textbox
             will open from which you can select a registered type
             library.

Examples:

  makepy.py -d

    Presents a list of registered type libraries from which you can make
    a selection.

  makepy.py -d "Microsoft Excel 8.0 Object Library"

    Generate support for the type library with the specified description
    (in this case, the MS Excel object model).

"""

import importlib
import os
import sys

import pythoncom
from win32com.client import Dispatch, gencache, genpy, selecttlb

bForDemandDefault = 0  # Default value of bForDemand - toggle this to change the world - see also gencache.py

error = "makepy.error"


def usage():
    sys.stderr.write(usageHelp)
    sys.exit(2)


def ShowInfo(spec):
    if not spec:
        tlbSpec = selecttlb.SelectTlb(excludeFlags=selecttlb.FLAG_HIDDEN)
        if tlbSpec is None:
            return
        try:
            tlb = pythoncom.LoadRegTypeLib(
                tlbSpec.clsid, tlbSpec.major, tlbSpec.minor, tlbSpec.lcid
            )
        except pythoncom.com_error:  # May be badly registered.
            sys.stderr.write(
                "Warning - could not load registered typelib '%s'\n" % (tlbSpec.clsid)
            )
            tlb = None

        infos = [(tlb, tlbSpec)]
    else:
        infos = GetTypeLibsForSpec(spec)
    for tlb, tlbSpec in infos:
        desc = tlbSpec.desc
        if desc is None:
            if tlb is None:
                desc = "<Could not load typelib %s>" % (tlbSpec.dll)
            else:
                desc = tlb.GetDocumentation(-1)[0]
        print(desc)
        print(
            " %s, lcid=%s, major=%s, minor=%s"
            % (tlbSpec.clsid, tlbSpec.lcid, tlbSpec.major, tlbSpec.minor)
        )
        print(" >>> # Use these commands in Python code to auto generate .py support")
        print(" >>> from win32com.client import gencache")
        print(
            " >>> gencache.EnsureModule('%s', %s, %s, %s)"
            % (tlbSpec.clsid, tlbSpec.lcid, tlbSpec.major, tlbSpec.minor)
        )


class SimpleProgress(genpy.GeneratorProgress):
    """A simple progress class prints its output to stderr"""

    def __init__(self, verboseLevel):
        self.verboseLevel = verboseLevel

    def Close(self):
        pass

    def Finished(self):
        if self.verboseLevel > 1:
            sys.stderr.write("Generation complete..\n")

    def SetDescription(self, desc, maxticks=None):
        if self.verboseLevel:
            sys.stderr.write(desc + "\n")

    def Tick(self, desc=None):
        pass

    def VerboseProgress(self, desc, verboseLevel=2):
        if self.verboseLevel >= verboseLevel:
            sys.stderr.write(desc + "\n")

    def LogBeginGenerate(self, filename):
        self.VerboseProgress("Generating to %s" % filename, 1)

    def LogWarning(self, desc):
        self.VerboseProgress("WARNING: " + desc, 1)


class GUIProgress(SimpleProgress):
    def __init__(self, verboseLevel):
        # Import some modules we need to we can trap failure now.
        import pywin  # nopycln: import
        import win32ui

        SimpleProgress.__init__(self, verboseLevel)
        self.dialog = None

    def Close(self):
        if self.dialog is not None:
            self.dialog.Close()
            self.dialog = None

    def Starting(self, tlb_desc):
        SimpleProgress.Starting(self, tlb_desc)
        if self.dialog is None:
            from pywin.dialogs import status

            self.dialog = status.ThreadedStatusProgressDialog(tlb_desc)
        else:
            self.dialog.SetTitle(tlb_desc)

    def SetDescription(self, desc, maxticks=None):
        self.dialog.SetText(desc)
        if maxticks:
            self.dialog.SetMaxTicks(maxticks)

    def Tick(self, desc=None):
        self.dialog.Tick()
        if desc is not None:
            self.dialog.SetText(desc)


def GetTypeLibsForSpec(arg):
    """Given an argument on the command line (either a file name, library
    description, or ProgID of an object) return a list of actual typelibs
    to use."""
    typelibs = []
    try:
        try:
            tlb = pythoncom.LoadTypeLib(arg)
            spec = selecttlb.TypelibSpec(None, 0, 0, 0)
            spec.FromTypelib(tlb, arg)
            typelibs.append((tlb, spec))
        except pythoncom.com_error:
            # See if it is a description
            tlbs = selecttlb.FindTlbsWithDescription(arg)
            if len(tlbs) == 0:
                # Maybe it is the name of a COM object?
                try:
                    ob = Dispatch(arg)
                    # and if so, it must support typelib info
                    tlb, index = ob._oleobj_.GetTypeInfo().GetContainingTypeLib()
                    spec = selecttlb.TypelibSpec(None, 0, 0, 0)
                    spec.FromTypelib(tlb)
                    tlbs.append(spec)
                except pythoncom.com_error:
                    pass
            if len(tlbs) == 0:
                print("Could not locate a type library matching '%s'" % (arg))
            for spec in tlbs:
                # Version numbers not always reliable if enumerated from registry.
                # (as some libs use hex, other's dont.  Both examples from MS, of course.)
                if spec.dll is None:
                    tlb = pythoncom.LoadRegTypeLib(
                        spec.clsid, spec.major, spec.minor, spec.lcid
                    )
                else:
                    tlb = pythoncom.LoadTypeLib(spec.dll)

                # We have a typelib, but it may not be exactly what we specified
                # (due to automatic version matching of COM).  So we query what we really have!
                attr = tlb.GetLibAttr()
                spec.major = attr[3]
                spec.minor = attr[4]
                spec.lcid = attr[1]
                typelibs.append((tlb, spec))
        return typelibs
    except pythoncom.com_error:
        t, v, tb = sys.exc_info()
        sys.stderr.write("Unable to load type library from '%s' - %s\n" % (arg, v))
        tb = None  # Storing tb in a local is a cycle!
        sys.exit(1)


def GenerateFromTypeLibSpec(
    typelibInfo,
    file=None,
    verboseLevel=None,
    progressInstance=None,
    bUnicodeToString=None,
    bForDemand=bForDemandDefault,
    bBuildHidden=1,
):
    assert bUnicodeToString is None, "this is deprecated and will go away"
    if verboseLevel is None:
        verboseLevel = 0  # By default, we use no gui and no verbose level!

    if bForDemand and file is not None:
        raise RuntimeError(
            "You can only perform a demand-build when the output goes to the gen_py directory"
        )
    if isinstance(typelibInfo, tuple):
        # Tuple
        typelibCLSID, lcid, major, minor = typelibInfo
        tlb = pythoncom.LoadRegTypeLib(typelibCLSID, major, minor, lcid)
        spec = selecttlb.TypelibSpec(typelibCLSID, lcid, major, minor)
        spec.FromTypelib(tlb, str(typelibCLSID))
        typelibs = [(tlb, spec)]
    elif isinstance(typelibInfo, selecttlb.TypelibSpec):
        if typelibInfo.dll is None:
            # Version numbers not always reliable if enumerated from registry.
            tlb = pythoncom.LoadRegTypeLib(
                typelibInfo.clsid,
                typelibInfo.major,
                typelibInfo.minor,
                typelibInfo.lcid,
            )
        else:
            tlb = pythoncom.LoadTypeLib(typelibInfo.dll)
        typelibs = [(tlb, typelibInfo)]
    elif hasattr(typelibInfo, "GetLibAttr"):
        # A real typelib object!
        # Could also use isinstance(typelibInfo, PyITypeLib) instead, but PyITypeLib is not directly exposed by pythoncom.
        # 	pythoncom.TypeIIDs[pythoncom.IID_ITypeLib] seems to work
        tla = typelibInfo.GetLibAttr()
        guid = tla[0]
        lcid = tla[1]
        major = tla[3]
        minor = tla[4]
        spec = selecttlb.TypelibSpec(guid, lcid, major, minor)
        typelibs = [(typelibInfo, spec)]
    else:
        typelibs = GetTypeLibsForSpec(typelibInfo)

    if progressInstance is None:
        progressInstance = SimpleProgress(verboseLevel)
    progress = progressInstance

    bToGenDir = file is None

    for typelib, info in typelibs:
        gen = genpy.Generator(typelib, info.dll, progress, bBuildHidden=bBuildHidden)

        if file is None:
            this_name = gencache.GetGeneratedFileName(
                info.clsid, info.lcid, info.major, info.minor
            )
            full_name = os.path.join(gencache.GetGeneratePath(), this_name)
            if bForDemand:
                try:
                    os.unlink(full_name + ".py")
                except os.error:
                    pass
                try:
                    os.unlink(full_name + ".pyc")
                except os.error:
                    pass
                try:
                    os.unlink(full_name + ".pyo")
                except os.error:
                    pass
                if not os.path.isdir(full_name):
                    os.mkdir(full_name)
                outputName = os.path.join(full_name, "__init__.py")
            else:
                outputName = full_name + ".py"
            fileUse = gen.open_writer(outputName)
            progress.LogBeginGenerate(outputName)
        else:
            fileUse = file

        worked = False
        try:
            gen.generate(fileUse, bForDemand)
            worked = True
        finally:
            if file is None:
                gen.finish_writer(outputName, fileUse, worked)
        importlib.invalidate_caches()
        if bToGenDir:
            progress.SetDescription("Importing module")
            gencache.AddModuleToCache(info.clsid, info.lcid, info.major, info.minor)

    progress.Close()


def GenerateChildFromTypeLibSpec(
    child, typelibInfo, verboseLevel=None, progressInstance=None, bUnicodeToString=None
):
    assert bUnicodeToString is None, "this is deprecated and will go away"
    if verboseLevel is None:
        verboseLevel = (
            0  # By default, we use no gui, and no verbose level for the children.
        )
    if type(typelibInfo) == type(()):
        typelibCLSID, lcid, major, minor = typelibInfo
        tlb = pythoncom.LoadRegTypeLib(typelibCLSID, major, minor, lcid)
    else:
        tlb = typelibInfo
        tla = typelibInfo.GetLibAttr()
        typelibCLSID = tla[0]
        lcid = tla[1]
        major = tla[3]
        minor = tla[4]
    spec = selecttlb.TypelibSpec(typelibCLSID, lcid, major, minor)
    spec.FromTypelib(tlb, str(typelibCLSID))
    typelibs = [(tlb, spec)]

    if progressInstance is None:
        progressInstance = SimpleProgress(verboseLevel)
    progress = progressInstance

    for typelib, info in typelibs:
        dir_name = gencache.GetGeneratedFileName(
            info.clsid, info.lcid, info.major, info.minor
        )
        dir_path_name = os.path.join(gencache.GetGeneratePath(), dir_name)
        progress.LogBeginGenerate(dir_path_name)

        gen = genpy.Generator(typelib, info.dll, progress)
        gen.generate_child(child, dir_path_name)
        progress.SetDescription("Importing module")
        importlib.invalidate_caches()
        __import__("win32com.gen_py." + dir_name + "." + child)
    progress.Close()


def main():
    import getopt

    hiddenSpec = 1
    outputName = None
    verboseLevel = 1
    doit = 1
    bForDemand = bForDemandDefault
    try:
        opts, args = getopt.getopt(sys.argv[1:], "vo:huiqd")
        for o, v in opts:
            if o == "-h":
                hiddenSpec = 0
            elif o == "-o":
                outputName = v
            elif o == "-v":
                verboseLevel = verboseLevel + 1
            elif o == "-q":
                verboseLevel = verboseLevel - 1
            elif o == "-i":
                if len(args) == 0:
                    ShowInfo(None)
                else:
                    for arg in args:
                        ShowInfo(arg)
                doit = 0
            elif o == "-d":
                bForDemand = not bForDemand

    except (getopt.error, error) as msg:
        sys.stderr.write(str(msg) + "\n")
        usage()

    if bForDemand and outputName is not None:
        sys.stderr.write("Can not use -d and -o together\n")
        usage()

    if not doit:
        return 0
    if len(args) == 0:
        rc = selecttlb.SelectTlb()
        if rc is None:
            sys.exit(1)
        args = [rc]

    if outputName is not None:
        path = os.path.dirname(outputName)
        if path != "" and not os.path.exists(path):
            os.makedirs(path)
        if sys.version_info > (3, 0):
            f = open(outputName, "wt", encoding="mbcs")
        else:
            import codecs  # not available in py3k.

            f = codecs.open(outputName, "w", "mbcs")
    else:
        f = None

    for arg in args:
        GenerateFromTypeLibSpec(
            arg,
            f,
            verboseLevel=verboseLevel,
            bForDemand=bForDemand,
            bBuildHidden=hiddenSpec,
        )

    if f:
        f.close()


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
    sys.exit(0)
