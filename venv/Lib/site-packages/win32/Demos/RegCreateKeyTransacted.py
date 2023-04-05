import win32api
import win32con
import win32transaction

keyname = "Pywin32 test transacted registry functions"
subkeyname = "test transacted subkey"
classname = "Transacted Class"

trans = win32transaction.CreateTransaction(Description="test RegCreateKeyTransacted")
key, disp = win32api.RegCreateKeyEx(
    win32con.HKEY_CURRENT_USER,
    keyname,
    samDesired=win32con.KEY_ALL_ACCESS,
    Class=classname,
)
## clean up any existing keys
for subk in win32api.RegEnumKeyExW(key):
    win32api.RegDeleteKey(key, subk[0])

## reopen key in transacted mode
transacted_key = win32api.RegOpenKeyTransacted(
    Key=win32con.HKEY_CURRENT_USER,
    SubKey=keyname,
    Transaction=trans,
    samDesired=win32con.KEY_ALL_ACCESS,
)
subkey, disp = win32api.RegCreateKeyEx(
    transacted_key,
    subkeyname,
    Transaction=trans,
    samDesired=win32con.KEY_ALL_ACCESS,
    Class=classname,
)

## Newly created key should not be visible from non-transacted handle
subkeys = [s[0] for s in win32api.RegEnumKeyExW(key)]
assert subkeyname not in subkeys

transacted_subkeys = [s[0] for s in win32api.RegEnumKeyExW(transacted_key)]
assert subkeyname in transacted_subkeys

## Key should be visible to non-transacted handle after commit
win32transaction.CommitTransaction(trans)
subkeys = [s[0] for s in win32api.RegEnumKeyExW(key)]
assert subkeyname in subkeys

## test transacted delete
del_trans = win32transaction.CreateTransaction(
    Description="test RegDeleteKeyTransacted"
)
win32api.RegDeleteKeyEx(key, subkeyname, Transaction=del_trans)
## subkey should still show up for non-transacted handle
subkeys = [s[0] for s in win32api.RegEnumKeyExW(key)]
assert subkeyname in subkeys
## ... and should be gone after commit
win32transaction.CommitTransaction(del_trans)
subkeys = [s[0] for s in win32api.RegEnumKeyExW(key)]
assert subkeyname not in subkeys

win32api.RegDeleteKey(win32con.HKEY_CURRENT_USER, keyname)
