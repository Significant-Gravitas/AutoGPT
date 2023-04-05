"""Simple base-classes for extensions and filters.

None of the filter and extension functions are considered 'optional' by the
framework.  These base-classes provide simple implementations for the
Initialize and Terminate functions, allowing you to omit them,

It is not necessary to use these base-classes - but if you don't, you
must ensure each of the required methods are implemented.
"""


class SimpleExtension:
    "Base class for a simple ISAPI extension"

    def __init__(self):
        pass

    def GetExtensionVersion(self, vi):
        """Called by the ISAPI framework to get the extension version

        The default implementation uses the classes docstring to
        set the extension description."""
        # nod to our reload capability - vi is None when we are reloaded.
        if vi is not None:
            vi.ExtensionDesc = self.__doc__

    def HttpExtensionProc(self, control_block):
        """Called by the ISAPI framework for each extension request.

        sub-classes must provide an implementation for this method.
        """
        raise NotImplementedError("sub-classes should override HttpExtensionProc")

    def TerminateExtension(self, status):
        """Called by the ISAPI framework as the extension terminates."""
        pass


class SimpleFilter:
    "Base class for a a simple ISAPI filter"
    filter_flags = None

    def __init__(self):
        pass

    def GetFilterVersion(self, fv):
        """Called by the ISAPI framework to get the extension version

        The default implementation uses the classes docstring to
        set the extension description, and uses the classes
        filter_flags attribute to set the ISAPI filter flags - you
        must specify filter_flags in your class.
        """
        if self.filter_flags is None:
            raise RuntimeError("You must specify the filter flags")
        # nod to our reload capability - fv is None when we are reloaded.
        if fv is not None:
            fv.Flags = self.filter_flags
            fv.FilterDesc = self.__doc__

    def HttpFilterProc(self, fc):
        """Called by the ISAPI framework for each filter request.

        sub-classes must provide an implementation for this method.
        """
        raise NotImplementedError("sub-classes should override HttpExtensionProc")

    def TerminateFilter(self, status):
        """Called by the ISAPI framework as the filter terminates."""
        pass
