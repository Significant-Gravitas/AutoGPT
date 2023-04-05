# is_platform_unicode is an old variable that was never correctly used and
# is no longer referenced in pywin32.  It is staying for a few releases incase
# others are looking at it, but it will go away soon!
is_platform_unicode = 0

# Ditto default_platform_encoding - not referenced and will die.
default_platform_encoding = "mbcs"

# This one *is* real and used - but in practice can't be changed.
default_scintilla_encoding = "utf-8"  # Scintilla _only_ supports this ATM
