# This is a python package
# __PackageSupportBuildPath__ not needed for distutil based builds,
# but not everyone is there yet.
import win32com

win32com.__PackageSupportBuildPath__(__path__)
