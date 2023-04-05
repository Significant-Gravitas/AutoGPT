# The Python ISAPI package.


# Exceptions thrown by the DLL framework.
class ISAPIError(Exception):
    def __init__(self, errno, strerror=None, funcname=None):
        # named attributes match IOError etc.
        self.errno = errno
        self.strerror = strerror
        self.funcname = funcname
        Exception.__init__(self, errno, strerror, funcname)

    def __str__(self):
        if self.strerror is None:
            try:
                import win32api

                self.strerror = win32api.FormatMessage(self.errno).strip()
            except:
                self.strerror = "no error message is available"
        # str() looks like a win32api error.
        return str((self.errno, self.strerror, self.funcname))


class FilterError(ISAPIError):
    pass


class ExtensionError(ISAPIError):
    pass


# A little development aid - a filter or extension callback function can
# raise one of these exceptions, and the handler module will be reloaded.
# This means you can change your code without restarting IIS.
# After a reload, your filter/extension will have the GetFilterVersion/
# GetExtensionVersion function called, but with None as the first arg.
class InternalReloadException(Exception):
    pass
