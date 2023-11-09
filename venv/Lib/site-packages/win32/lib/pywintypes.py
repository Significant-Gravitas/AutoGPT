# Magic utility that "redirects" to pywintypesxx.dll
import importlib.machinery
import importlib.util
import os
import sys


def __import_pywin32_system_module__(modname, globs):
    # This has been through a number of iterations.  The problem: how to
    # locate pywintypesXX.dll when it may be in a number of places, and how
    # to avoid ever loading it twice.  This problem is compounded by the
    # fact that the "right" way to do this requires win32api, but this
    # itself requires pywintypesXX.
    # And the killer problem is that someone may have done 'import win32api'
    # before this code is called.  In that case Windows will have already
    # loaded pywintypesXX as part of loading win32api - but by the time
    # we get here, we may locate a different one.  This appears to work, but
    # then starts raising bizarre TypeErrors complaining that something
    # is not a pywintypes type when it clearly is!

    # So in what we hope is the last major iteration of this, we now
    # rely on a _win32sysloader module, implemented in C but not relying
    # on pywintypesXX.dll.  It then can check if the DLL we are looking for
    # lib is already loaded.
    # See if this is a debug build.
    suffix = "_d" if "_d.pyd" in importlib.machinery.EXTENSION_SUFFIXES else ""
    filename = "%s%d%d%s.dll" % (
        modname,
        sys.version_info[0],
        sys.version_info[1],
        suffix,
    )
    if hasattr(sys, "frozen"):
        # If we are running from a frozen program (py2exe, McMillan, freeze)
        # then we try and load the DLL from our sys.path
        # XXX - This path may also benefit from _win32sysloader?  However,
        # MarkH has never seen the DLL load problem with py2exe programs...
        for look in sys.path:
            # If the sys.path entry is a (presumably) .zip file, use the
            # directory
            if os.path.isfile(look):
                look = os.path.dirname(look)
            found = os.path.join(look, filename)
            if os.path.isfile(found):
                break
        else:
            raise ImportError(
                "Module '%s' isn't in frozen sys.path %s" % (modname, sys.path)
            )
    else:
        # First see if it already in our process - if so, we must use that.
        import _win32sysloader

        found = _win32sysloader.GetModuleFilename(filename)
        if found is None:
            # We ask Windows to load it next.  This is in an attempt to
            # get the exact same module loaded should pywintypes be imported
            # first (which is how we are here) or if, eg, win32api was imported
            # first thereby implicitly loading the DLL.

            # Sadly though, it doesn't quite work - if pywintypesxx.dll
            # is in system32 *and* the executable's directory, on XP SP2, an
            # import of win32api will cause Windows to load pywintypes
            # from system32, where LoadLibrary for that name will
            # load the one in the exe's dir.
            # That shouldn't really matter though, so long as we only ever
            # get one loaded.
            found = _win32sysloader.LoadModule(filename)
        if found is None:
            # Windows can't find it - which although isn't relevent here,
            # means that we *must* be the first win32 import, as an attempt
            # to import win32api etc would fail when Windows attempts to
            # locate the DLL.
            # This is most likely to happen for "non-admin" installs, where
            # we can't put the files anywhere else on the global path.

            # If there is a version in our Python directory, use that
            if os.path.isfile(os.path.join(sys.prefix, filename)):
                found = os.path.join(sys.prefix, filename)
        if found is None:
            # Not in the Python directory?  Maybe we were installed via
            # easy_install...
            if os.path.isfile(os.path.join(os.path.dirname(__file__), filename)):
                found = os.path.join(os.path.dirname(__file__), filename)

        # There are 2 site-packages directories - one "global" and one "user".
        # We could be in either, or both (but with different versions!). Factors include
        # virtualenvs, post-install script being run or not, `setup.py install` flags, etc.

        # In a worst-case, it means, say 'python -c "import win32api"'
        # will not work but 'python -c "import pywintypes, win32api"' will,
        # but it's better than nothing.

        # We use the same logic as pywin32_bootstrap to find potential location for the dll
        # Simply import pywin32_system32 and look in the paths in pywin32_system32.__path__

        if found is None:
            import pywin32_system32

            for path in pywin32_system32.__path__:
                maybe = os.path.join(path, filename)
                if os.path.isfile(maybe):
                    found = maybe
                    break

        if found is None:
            # give up in disgust.
            raise ImportError("No system module '%s' (%s)" % (modname, filename))
    # After importing the module, sys.modules is updated to the DLL we just
    # loaded - which isn't what we want. So we update sys.modules to refer to
    # this module, and update our globals from it.
    old_mod = sys.modules[modname]
    # Load the DLL.
    loader = importlib.machinery.ExtensionFileLoader(modname, found)
    spec = importlib.machinery.ModuleSpec(name=modname, loader=loader, origin=found)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Check the sys.modules[] behaviour we describe above is true...
    assert sys.modules[modname] is mod
    # as above - re-reset to the *old* module object then update globs.
    sys.modules[modname] = old_mod
    globs.update(mod.__dict__)


__import_pywin32_system_module__("pywintypes", globals())
