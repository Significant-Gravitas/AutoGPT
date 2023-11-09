# __init__ for the Pythonwin editor package.
#
# We used to support optional editors - eg, color or non-color.
#
# This really isnt necessary with Scintilla, and scintilla
# is getting so deeply embedded that it was too much work.

import sys

import win32con
import win32ui

defaultCharacterFormat = (-402653169, 0, 200, 0, 0, 0, 49, "Courier New")

##def GetDefaultEditorModuleName():
##	import pywin
##	# If someone has set pywin.editormodulename, then this is what we use
##	try:
##		prefModule = pywin.editormodulename
##	except AttributeError:
##		prefModule = win32ui.GetProfileVal("Editor","Module", "")
##	return prefModule
##
##def WriteDefaultEditorModule(module):
##	try:
##		module = module.__name__
##	except:
##		pass
##	win32ui.WriteProfileVal("Editor", "Module", module)


def LoadDefaultEditor():
    pass


##	prefModule = GetDefaultEditorModuleName()
##	restorePrefModule = None
##	mod = None
##	if prefModule:
##		try:
##			mod = __import__(prefModule)
##		except 'xx':
##			msg = "Importing your preferred editor ('%s') failed.\n\nError %s: %s\n\nAn attempt will be made to load the default editor.\n\nWould you like this editor disabled in the future?" % (prefModule, sys.exc_info()[0], sys.exc_info()[1])
##			rc = win32ui.MessageBox(msg, "Error importing editor", win32con.MB_YESNO)
##			if rc == win32con.IDNO:
##				restorePrefModule = prefModule
##			WriteDefaultEditorModule("")
##			del rc
##
##	try:
##		# Try and load the default one - dont catch errors here.
##		if mod is None:
##			prefModule = "pywin.framework.editor.color.coloreditor"
##			mod = __import__(prefModule)
##
##		# Get at the real module.
##		mod = sys.modules[prefModule]
##
##		# Do a "from mod import *"
##		globals().update(mod.__dict__)
##
##	finally:
##		# Restore the users default editor if it failed and they requested not to disable it.
##		if restorePrefModule:
##			WriteDefaultEditorModule(restorePrefModule)


def GetEditorOption(option, defaultValue, min=None, max=None):
    rc = win32ui.GetProfileVal("Editor", option, defaultValue)
    if min is not None and rc < min:
        rc = defaultValue
    if max is not None and rc > max:
        rc = defaultValue
    return rc


def SetEditorOption(option, newValue):
    win32ui.WriteProfileVal("Editor", option, newValue)


def DeleteEditorOption(option):
    try:
        win32ui.WriteProfileVal("Editor", option, None)
    except win32ui.error:
        pass


# Load and save font tuples
def GetEditorFontOption(option, default=None):
    if default is None:
        default = defaultCharacterFormat
    fmt = GetEditorOption(option, "")
    if fmt == "":
        return default
    try:
        return eval(fmt)
    except:
        print("WARNING: Invalid font setting in registry - setting ignored")
        return default


def SetEditorFontOption(option, newValue):
    SetEditorOption(option, str(newValue))


from pywin.framework.editor.color.coloreditor import editorTemplate
