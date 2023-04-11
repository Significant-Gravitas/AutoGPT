"""setuptools.errors

Provides exceptions used by setuptools modules.
"""

from distutils import errors as _distutils_errors
from distutils.errors import DistutilsError


class RemovedCommandError(DistutilsError, RuntimeError):
    """Error used for commands that have been removed in setuptools.

    Since ``setuptools`` is built on ``distutils``, simply removing a command
    from ``setuptools`` will make the behavior fall back to ``distutils``; this
    error is raised if a command exists in ``distutils`` but has been actively
    removed in ``setuptools``.
    """


# Re-export errors from distutils to facilitate the migration to PEP632

ByteCompileError = _distutils_errors.DistutilsByteCompileError
CCompilerError = _distutils_errors.CCompilerError
ClassError = _distutils_errors.DistutilsClassError
CompileError = _distutils_errors.CompileError
ExecError = _distutils_errors.DistutilsExecError
FileError = _distutils_errors.DistutilsFileError
InternalError = _distutils_errors.DistutilsInternalError
LibError = _distutils_errors.LibError
LinkError = _distutils_errors.LinkError
ModuleError = _distutils_errors.DistutilsModuleError
OptionError = _distutils_errors.DistutilsOptionError
PlatformError = _distutils_errors.DistutilsPlatformError
PreprocessError = _distutils_errors.PreprocessError
SetupError = _distutils_errors.DistutilsSetupError
TemplateError = _distutils_errors.DistutilsTemplateError
UnknownFileError = _distutils_errors.UnknownFileError

# The root error class in the hierarchy
BaseError = _distutils_errors.DistutilsError
