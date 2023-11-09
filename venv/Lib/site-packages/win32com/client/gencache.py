"""Manages the cache of generated Python code.

Description
  This file manages the cache of generated Python code.  When run from the
  command line, it also provides a number of options for managing that cache.

Implementation
  Each typelib is generated into a filename of format "{guid}x{lcid}x{major}x{minor}.py"

  An external persistant dictionary maps from all known IIDs in all known type libraries
  to the type library itself.

  Thus, whenever Python code knows the IID of an object, it can find the IID, LCID and version of
  the type library which supports it.  Given this information, it can find the Python module
  with the support.

  If necessary, this support can be generated on the fly.

Hacks, to do, etc
  Currently just uses a pickled dictionary, but should used some sort of indexed file.
  Maybe an OLE2 compound file, or a bsddb file?
"""
import glob
import os
import sys
from importlib import reload

import pythoncom
import pywintypes
import win32com
import win32com.client

from . import CLSIDToClass

bForDemandDefault = 0  # Default value of bForDemand - toggle this to change the world - see also makepy.py

# The global dictionary
clsidToTypelib = {}

# If we have a different version of the typelib generated, this
# maps the "requested version" to the "generated version".
versionRedirectMap = {}

# There is no reason we *must* be readonly in a .zip, but we are now,
# Rather than check for ".zip" or other tricks, PEP302 defines
# a "__loader__" attribute, so we use that.
# (Later, it may become necessary to check if the __loader__ can update files,
# as a .zip loader potentially could - but punt all that until a need arises)
is_readonly = is_zip = hasattr(win32com, "__loader__") and hasattr(
    win32com.__loader__, "archive"
)

# A dictionary of ITypeLibrary objects for demand generation explicitly handed to us
# Keyed by usual clsid, lcid, major, minor
demandGeneratedTypeLibraries = {}

import pickle as pickle


def __init__():
    # Initialize the module.  Called once explicitly at module import below.
    try:
        _LoadDicts()
    except IOError:
        Rebuild()


pickleVersion = 1


def _SaveDicts():
    if is_readonly:
        raise RuntimeError(
            "Trying to write to a readonly gencache ('%s')!" % win32com.__gen_path__
        )
    f = open(os.path.join(GetGeneratePath(), "dicts.dat"), "wb")
    try:
        p = pickle.Pickler(f)
        p.dump(pickleVersion)
        p.dump(clsidToTypelib)
    finally:
        f.close()


def _LoadDicts():
    # Load the dictionary from a .zip file if that is where we live.
    if is_zip:
        import io as io

        loader = win32com.__loader__
        arc_path = loader.archive
        dicts_path = os.path.join(win32com.__gen_path__, "dicts.dat")
        if dicts_path.startswith(arc_path):
            dicts_path = dicts_path[len(arc_path) + 1 :]
        else:
            # Hm. See below.
            return
        try:
            data = loader.get_data(dicts_path)
        except AttributeError:
            # The __loader__ has no get_data method.  See below.
            return
        except IOError:
            # Our gencache is in a .zip file (and almost certainly readonly)
            # but no dicts file.  That actually needn't be fatal for a frozen
            # application.  Assuming they call "EnsureModule" with the same
            # typelib IDs they have been frozen with, that EnsureModule will
            # correctly re-build the dicts on the fly.  However, objects that
            # rely on the gencache but have not done an EnsureModule will
            # fail (but their apps are likely to fail running from source
            # with a clean gencache anyway, as then they would be getting
            # Dynamic objects until the cache is built - so the best answer
            # for these apps is to call EnsureModule, rather than freezing
            # the dict)
            return
        f = io.BytesIO(data)
    else:
        # NOTE: IOError on file open must be caught by caller.
        f = open(os.path.join(win32com.__gen_path__, "dicts.dat"), "rb")
    try:
        p = pickle.Unpickler(f)
        version = p.load()
        global clsidToTypelib
        clsidToTypelib = p.load()
        versionRedirectMap.clear()
    finally:
        f.close()


def GetGeneratedFileName(clsid, lcid, major, minor):
    """Given the clsid, lcid, major and  minor for a type lib, return
    the file name (no extension) providing this support.
    """
    return str(clsid).upper()[1:-1] + "x%sx%sx%s" % (lcid, major, minor)


def SplitGeneratedFileName(fname):
    """Reverse of GetGeneratedFileName()"""
    return tuple(fname.split("x", 4))


def GetGeneratePath():
    """Returns the name of the path to generate to.
    Checks the directory is OK.
    """
    assert not is_readonly, "Why do you want the genpath for a readonly store?"
    try:
        os.makedirs(win32com.__gen_path__)
        # os.mkdir(win32com.__gen_path__)
    except os.error:
        pass
    try:
        fname = os.path.join(win32com.__gen_path__, "__init__.py")
        os.stat(fname)
    except os.error:
        f = open(fname, "w")
        f.write(
            "# Generated file - this directory may be deleted to reset the COM cache...\n"
        )
        f.write("import win32com\n")
        f.write(
            "if __path__[:-1] != win32com.__gen_path__: __path__.append(win32com.__gen_path__)\n"
        )
        f.close()

    return win32com.__gen_path__


#
# The helpers for win32com.client.Dispatch and OCX clients.
#
def GetClassForProgID(progid):
    """Get a Python class for a Program ID

    Given a Program ID, return a Python class which wraps the COM object

    Returns the Python class, or None if no module is available.

    Params
    progid -- A COM ProgramID or IID (eg, "Word.Application")
    """
    clsid = pywintypes.IID(progid)  # This auto-converts named to IDs.
    return GetClassForCLSID(clsid)


def GetClassForCLSID(clsid):
    """Get a Python class for a CLSID

    Given a CLSID, return a Python class which wraps the COM object

    Returns the Python class, or None if no module is available.

    Params
    clsid -- A COM CLSID (or string repr of one)
    """
    # first, take a short-cut - we may already have generated support ready-to-roll.
    clsid = str(clsid)
    if CLSIDToClass.HasClass(clsid):
        return CLSIDToClass.GetClass(clsid)
    mod = GetModuleForCLSID(clsid)
    if mod is None:
        return None
    try:
        return CLSIDToClass.GetClass(clsid)
    except KeyError:
        return None


def GetModuleForProgID(progid):
    """Get a Python module for a Program ID

    Given a Program ID, return a Python module which contains the
    class which wraps the COM object.

    Returns the Python module, or None if no module is available.

    Params
    progid -- A COM ProgramID or IID (eg, "Word.Application")
    """
    try:
        iid = pywintypes.IID(progid)
    except pywintypes.com_error:
        return None
    return GetModuleForCLSID(iid)


def GetModuleForCLSID(clsid):
    """Get a Python module for a CLSID

    Given a CLSID, return a Python module which contains the
    class which wraps the COM object.

    Returns the Python module, or None if no module is available.

    Params
    progid -- A COM CLSID (ie, not the description)
    """
    clsid_str = str(clsid)
    try:
        typelibCLSID, lcid, major, minor = clsidToTypelib[clsid_str]
    except KeyError:
        return None

    try:
        mod = GetModuleForTypelib(typelibCLSID, lcid, major, minor)
    except ImportError:
        mod = None
    if mod is not None:
        sub_mod = mod.CLSIDToPackageMap.get(clsid_str)
        if sub_mod is None:
            sub_mod = mod.VTablesToPackageMap.get(clsid_str)
        if sub_mod is not None:
            sub_mod_name = mod.__name__ + "." + sub_mod
            try:
                __import__(sub_mod_name)
            except ImportError:
                info = typelibCLSID, lcid, major, minor
                # Force the generation.  If this typelibrary has explicitly been added,
                # use it (it may not be registered, causing a lookup by clsid to fail)
                if info in demandGeneratedTypeLibraries:
                    info = demandGeneratedTypeLibraries[info]
                from . import makepy

                makepy.GenerateChildFromTypeLibSpec(sub_mod, info)
                # Generate does an import...
            mod = sys.modules[sub_mod_name]
    return mod


def GetModuleForTypelib(typelibCLSID, lcid, major, minor):
    """Get a Python module for a type library ID

    Given the CLSID of a typelibrary, return an imported Python module,
    else None

    Params
    typelibCLSID -- IID of the type library.
    major -- Integer major version.
    minor -- Integer minor version
    lcid -- Integer LCID for the library.
    """
    modName = GetGeneratedFileName(typelibCLSID, lcid, major, minor)
    mod = _GetModule(modName)
    # If the import worked, it doesn't mean we have actually added this
    # module to our cache though - check that here.
    if "_in_gencache_" not in mod.__dict__:
        AddModuleToCache(typelibCLSID, lcid, major, minor)
        assert "_in_gencache_" in mod.__dict__
    return mod


def MakeModuleForTypelib(
    typelibCLSID,
    lcid,
    major,
    minor,
    progressInstance=None,
    bForDemand=bForDemandDefault,
    bBuildHidden=1,
):
    """Generate support for a type library.

    Given the IID, LCID and version information for a type library, generate
    and import the necessary support files.

    Returns the Python module.  No exceptions are caught.

    Params
    typelibCLSID -- IID of the type library.
    major -- Integer major version.
    minor -- Integer minor version.
    lcid -- Integer LCID for the library.
    progressInstance -- Instance to use as progress indicator, or None to
                        use the GUI progress bar.
    """
    from . import makepy

    makepy.GenerateFromTypeLibSpec(
        (typelibCLSID, lcid, major, minor),
        progressInstance=progressInstance,
        bForDemand=bForDemand,
        bBuildHidden=bBuildHidden,
    )
    return GetModuleForTypelib(typelibCLSID, lcid, major, minor)


def MakeModuleForTypelibInterface(
    typelib_ob, progressInstance=None, bForDemand=bForDemandDefault, bBuildHidden=1
):
    """Generate support for a type library.

    Given a PyITypeLib interface generate and import the necessary support files.  This is useful
    for getting makepy support for a typelibrary that is not registered - the caller can locate
    and load the type library itself, rather than relying on COM to find it.

    Returns the Python module.

    Params
    typelib_ob -- The type library itself
    progressInstance -- Instance to use as progress indicator, or None to
                        use the GUI progress bar.
    """
    from . import makepy

    try:
        makepy.GenerateFromTypeLibSpec(
            typelib_ob,
            progressInstance=progressInstance,
            bForDemand=bForDemandDefault,
            bBuildHidden=bBuildHidden,
        )
    except pywintypes.com_error:
        return None
    tla = typelib_ob.GetLibAttr()
    guid = tla[0]
    lcid = tla[1]
    major = tla[3]
    minor = tla[4]
    return GetModuleForTypelib(guid, lcid, major, minor)


def EnsureModuleForTypelibInterface(
    typelib_ob, progressInstance=None, bForDemand=bForDemandDefault, bBuildHidden=1
):
    """Check we have support for a type library, generating if not.

    Given a PyITypeLib interface generate and import the necessary
    support files if necessary. This is useful for getting makepy support
    for a typelibrary that is not registered - the caller can locate and
    load the type library itself, rather than relying on COM to find it.

    Returns the Python module.

    Params
    typelib_ob -- The type library itself
    progressInstance -- Instance to use as progress indicator, or None to
                        use the GUI progress bar.
    """
    tla = typelib_ob.GetLibAttr()
    guid = tla[0]
    lcid = tla[1]
    major = tla[3]
    minor = tla[4]

    # If demand generated, save the typelib interface away for later use
    if bForDemand:
        demandGeneratedTypeLibraries[(str(guid), lcid, major, minor)] = typelib_ob

    try:
        return GetModuleForTypelib(guid, lcid, major, minor)
    except ImportError:
        pass
    # Generate it.
    return MakeModuleForTypelibInterface(
        typelib_ob, progressInstance, bForDemand, bBuildHidden
    )


def ForgetAboutTypelibInterface(typelib_ob):
    """Drop any references to a typelib previously added with EnsureModuleForTypelibInterface and forDemand"""
    tla = typelib_ob.GetLibAttr()
    guid = tla[0]
    lcid = tla[1]
    major = tla[3]
    minor = tla[4]
    info = str(guid), lcid, major, minor
    try:
        del demandGeneratedTypeLibraries[info]
    except KeyError:
        # Not worth raising an exception - maybe they dont know we only remember for demand generated, etc.
        print(
            "ForgetAboutTypelibInterface:: Warning - type library with info %s is not being remembered!"
            % (info,)
        )
    # and drop any version redirects to it
    for key, val in list(versionRedirectMap.items()):
        if val == info:
            del versionRedirectMap[key]


def EnsureModule(
    typelibCLSID,
    lcid,
    major,
    minor,
    progressInstance=None,
    bValidateFile=not is_readonly,
    bForDemand=bForDemandDefault,
    bBuildHidden=1,
):
    """Ensure Python support is loaded for a type library, generating if necessary.

    Given the IID, LCID and version information for a type library, check and if
    necessary (re)generate, then import the necessary support files. If we regenerate the file, there
    is no way to totally snuff out all instances of the old module in Python, and thus we will regenerate the file more than necessary,
    unless makepy/genpy is modified accordingly.


    Returns the Python module.  No exceptions are caught during the generate process.

    Params
    typelibCLSID -- IID of the type library.
    major -- Integer major version.
    minor -- Integer minor version
    lcid -- Integer LCID for the library.
    progressInstance -- Instance to use as progress indicator, or None to
                        use the GUI progress bar.
    bValidateFile -- Whether or not to perform cache validation or not
    bForDemand -- Should a complete generation happen now, or on demand?
    bBuildHidden -- Should hidden members/attributes etc be generated?
    """
    bReloadNeeded = 0
    try:
        try:
            module = GetModuleForTypelib(typelibCLSID, lcid, major, minor)
        except ImportError:
            # If we get an ImportError
            # We may still find a valid cache file under a different MinorVersion #
            # (which windows will search out for us)
            # print "Loading reg typelib", typelibCLSID, major, minor, lcid
            module = None
            try:
                tlbAttr = pythoncom.LoadRegTypeLib(
                    typelibCLSID, major, minor, lcid
                ).GetLibAttr()
                # if the above line doesn't throw a pythoncom.com_error, check if
                # it is actually a different lib than we requested, and if so, suck it in
                if tlbAttr[1] != lcid or tlbAttr[4] != minor:
                    # print "Trying 2nd minor #", tlbAttr[1], tlbAttr[3], tlbAttr[4]
                    try:
                        module = GetModuleForTypelib(
                            typelibCLSID, tlbAttr[1], tlbAttr[3], tlbAttr[4]
                        )
                    except ImportError:
                        # We don't have a module, but we do have a better minor
                        # version - remember that.
                        minor = tlbAttr[4]
                # else module remains None
            except pythoncom.com_error:
                # couldn't load any typelib - mod remains None
                pass
        if module is not None and bValidateFile:
            assert not is_readonly, "Can't validate in a read-only gencache"
            try:
                typLibPath = pythoncom.QueryPathOfRegTypeLib(
                    typelibCLSID, major, minor, lcid
                )
                # windows seems to add an extra \0 (via the underlying BSTR)
                # The mainwin toolkit does not add this erroneous \0
                if typLibPath[-1] == "\0":
                    typLibPath = typLibPath[:-1]
                suf = getattr(os.path, "supports_unicode_filenames", 0)
                if not suf:
                    # can't pass unicode filenames directly - convert
                    try:
                        typLibPath = typLibPath.encode(sys.getfilesystemencoding())
                    except AttributeError:  # no sys.getfilesystemencoding
                        typLibPath = str(typLibPath)
                tlbAttributes = pythoncom.LoadRegTypeLib(
                    typelibCLSID, major, minor, lcid
                ).GetLibAttr()
            except pythoncom.com_error:
                # We have a module, but no type lib - we should still
                # run with what we have though - the typelib may not be
                # deployed here.
                bValidateFile = 0
        if module is not None and bValidateFile:
            assert not is_readonly, "Can't validate in a read-only gencache"
            filePathPrefix = "%s\\%s" % (
                GetGeneratePath(),
                GetGeneratedFileName(typelibCLSID, lcid, major, minor),
            )
            filePath = filePathPrefix + ".py"
            filePathPyc = filePathPrefix + ".py"
            if __debug__:
                filePathPyc = filePathPyc + "c"
            else:
                filePathPyc = filePathPyc + "o"
            # Verify that type library is up to date.
            # If we have a differing MinorVersion or genpy has bumped versions, update the file
            from . import genpy

            if (
                module.MinorVersion != tlbAttributes[4]
                or genpy.makepy_version != module.makepy_version
            ):
                # print "Version skew: %d, %d" % (module.MinorVersion, tlbAttributes[4])
                # try to erase the bad file from the cache
                try:
                    os.unlink(filePath)
                except os.error:
                    pass
                try:
                    os.unlink(filePathPyc)
                except os.error:
                    pass
                if os.path.isdir(filePathPrefix):
                    import shutil

                    shutil.rmtree(filePathPrefix)
                minor = tlbAttributes[4]
                module = None
                bReloadNeeded = 1
            else:
                minor = module.MinorVersion
                filePathPrefix = "%s\\%s" % (
                    GetGeneratePath(),
                    GetGeneratedFileName(typelibCLSID, lcid, major, minor),
                )
                filePath = filePathPrefix + ".py"
                filePathPyc = filePathPrefix + ".pyc"
                # print "Trying py stat: ", filePath
                fModTimeSet = 0
                try:
                    pyModTime = os.stat(filePath)[8]
                    fModTimeSet = 1
                except os.error as e:
                    # If .py file fails, try .pyc file
                    # print "Trying pyc stat", filePathPyc
                    try:
                        pyModTime = os.stat(filePathPyc)[8]
                        fModTimeSet = 1
                    except os.error as e:
                        pass
                # print "Trying stat typelib", pyModTime
                # print str(typLibPath)
                typLibModTime = os.stat(typLibPath)[8]
                if fModTimeSet and (typLibModTime > pyModTime):
                    bReloadNeeded = 1
                    module = None
    except (ImportError, os.error):
        module = None
    if module is None:
        # We need to build an item.  If we are in a read-only cache, we
        # can't/don't want to do this - so before giving up, check for
        # a different minor version in our cache - according to COM, this is OK
        if is_readonly:
            key = str(typelibCLSID), lcid, major, minor
            # If we have been asked before, get last result.
            try:
                return versionRedirectMap[key]
            except KeyError:
                pass
            # Find other candidates.
            items = []
            for desc in GetGeneratedInfos():
                if key[0] == desc[0] and key[1] == desc[1] and key[2] == desc[2]:
                    items.append(desc)
            if items:
                # Items are all identical, except for last tuple element
                # We want the latest minor version we have - so just sort and grab last
                items.sort()
                new_minor = items[-1][3]
                ret = GetModuleForTypelib(typelibCLSID, lcid, major, new_minor)
            else:
                ret = None
            # remember and return
            versionRedirectMap[key] = ret
            return ret
        # print "Rebuilding: ", major, minor
        module = MakeModuleForTypelib(
            typelibCLSID,
            lcid,
            major,
            minor,
            progressInstance,
            bForDemand=bForDemand,
            bBuildHidden=bBuildHidden,
        )
        # If we replaced something, reload it
        if bReloadNeeded:
            module = reload(module)
            AddModuleToCache(typelibCLSID, lcid, major, minor)
    return module


def EnsureDispatch(
    prog_id, bForDemand=1
):  # New fn, so we default the new demand feature to on!
    """Given a COM prog_id, return an object that is using makepy support, building if necessary"""
    disp = win32com.client.Dispatch(prog_id)
    if not disp.__dict__.get("CLSID"):  # Eeek - no makepy support - try and build it.
        try:
            ti = disp._oleobj_.GetTypeInfo()
            disp_clsid = ti.GetTypeAttr()[0]
            tlb, index = ti.GetContainingTypeLib()
            tla = tlb.GetLibAttr()
            mod = EnsureModule(tla[0], tla[1], tla[3], tla[4], bForDemand=bForDemand)
            GetModuleForCLSID(disp_clsid)
            # Get the class from the module.
            from . import CLSIDToClass

            disp_class = CLSIDToClass.GetClass(str(disp_clsid))
            disp = disp_class(disp._oleobj_)
        except pythoncom.com_error:
            raise TypeError(
                "This COM object can not automate the makepy process - please run makepy manually for this object"
            )
    return disp


def AddModuleToCache(
    typelibclsid, lcid, major, minor, verbose=1, bFlushNow=not is_readonly
):
    """Add a newly generated file to the cache dictionary."""
    fname = GetGeneratedFileName(typelibclsid, lcid, major, minor)
    mod = _GetModule(fname)
    # if mod._in_gencache_ is already true, then we are reloading this
    # module - this doesn't mean anything special though!
    mod._in_gencache_ = 1
    info = str(typelibclsid), lcid, major, minor
    dict_modified = False

    def SetTypelibForAllClsids(dict):
        nonlocal dict_modified
        for clsid, cls in dict.items():
            if clsidToTypelib.get(clsid) != info:
                clsidToTypelib[clsid] = info
                dict_modified = True

    SetTypelibForAllClsids(mod.CLSIDToClassMap)
    SetTypelibForAllClsids(mod.CLSIDToPackageMap)
    SetTypelibForAllClsids(mod.VTablesToClassMap)
    SetTypelibForAllClsids(mod.VTablesToPackageMap)

    # If this lib was previously redirected, drop it
    if info in versionRedirectMap:
        del versionRedirectMap[info]
    if bFlushNow and dict_modified:
        _SaveDicts()


def GetGeneratedInfos():
    zip_pos = win32com.__gen_path__.find(".zip\\")
    if zip_pos >= 0:
        import zipfile

        zip_file = win32com.__gen_path__[: zip_pos + 4]
        zip_path = win32com.__gen_path__[zip_pos + 5 :].replace("\\", "/")
        zf = zipfile.ZipFile(zip_file)
        infos = {}
        for n in zf.namelist():
            if not n.startswith(zip_path):
                continue
            base = n[len(zip_path) + 1 :].split("/")[0]
            try:
                iid, lcid, major, minor = base.split("x")
                lcid = int(lcid)
                major = int(major)
                minor = int(minor)
                iid = pywintypes.IID("{" + iid + "}")
            except ValueError:
                continue
            except pywintypes.com_error:
                # invalid IID
                continue
            infos[(iid, lcid, major, minor)] = 1
        zf.close()
        return list(infos.keys())
    else:
        # on the file system
        files = glob.glob(win32com.__gen_path__ + "\\*")
        ret = []
        for file in files:
            if not os.path.isdir(file) and not os.path.splitext(file)[1] == ".py":
                continue
            name = os.path.splitext(os.path.split(file)[1])[0]
            try:
                iid, lcid, major, minor = name.split("x")
                iid = pywintypes.IID("{" + iid + "}")
                lcid = int(lcid)
                major = int(major)
                minor = int(minor)
            except ValueError:
                continue
            except pywintypes.com_error:
                # invalid IID
                continue
            ret.append((iid, lcid, major, minor))
        return ret


def _GetModule(fname):
    """Given the name of a module in the gen_py directory, import and return it."""
    mod_name = "win32com.gen_py.%s" % fname
    mod = __import__(mod_name)
    return sys.modules[mod_name]


def Rebuild(verbose=1):
    """Rebuild the cache indexes from the file system."""
    clsidToTypelib.clear()
    infos = GetGeneratedInfos()
    if verbose and len(infos):  # Dont bother reporting this when directory is empty!
        print("Rebuilding cache of generated files for COM support...")
    for info in infos:
        iid, lcid, major, minor = info
        if verbose:
            print("Checking", GetGeneratedFileName(*info))
        try:
            AddModuleToCache(iid, lcid, major, minor, verbose, 0)
        except:
            print(
                "Could not add module %s - %s: %s"
                % (info, sys.exc_info()[0], sys.exc_info()[1])
            )
    if verbose and len(infos):  # Dont bother reporting this when directory is empty!
        print("Done.")
    _SaveDicts()


def _Dump():
    print("Cache is in directory", win32com.__gen_path__)
    # Build a unique dir
    d = {}
    for clsid, (typelibCLSID, lcid, major, minor) in clsidToTypelib.items():
        d[typelibCLSID, lcid, major, minor] = None
    for typelibCLSID, lcid, major, minor in d.keys():
        mod = GetModuleForTypelib(typelibCLSID, lcid, major, minor)
        print("%s - %s" % (mod.__doc__, typelibCLSID))


# Boot up
__init__()


def usage():
    usageString = """\
	  Usage: gencache [-q] [-d] [-r]

			 -q         - Quiet
			 -d         - Dump the cache (typelibrary description and filename).
			 -r         - Rebuild the cache dictionary from the existing .py files
	"""
    print(usageString)
    sys.exit(1)


if __name__ == "__main__":
    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "qrd")
    except getopt.error as message:
        print(message)
        usage()

    # we only have options - complain about real args, or none at all!
    if len(sys.argv) == 1 or args:
        print(usage())

    verbose = 1
    for opt, val in opts:
        if opt == "-d":  # Dump
            _Dump()
        if opt == "-r":
            Rebuild(verbose)
        if opt == "-q":
            verbose = 0
