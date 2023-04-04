# win2kras used to be an extension module with wrapped the "new" RAS functions
# in Windows 2000, so win32ras could still be used on NT/etc.
# I think in 2021 we can be confident pywin32 is not used on earlier OSs, so
# that functionality is now in win32ras.
#
# This exists just to avoid breaking old scripts.
from win32ras import *
