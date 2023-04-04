# See if we have a special directory for the binaries (for developers)
import win32com

win32com.__PackageSupportBuildPath__(__path__)
