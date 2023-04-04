import win32security

policy_handle = win32security.GetPolicyHandle("", win32security.POLICY_ALL_ACCESS)
privatedata = "some sensitive data"
keyname = "tmp"
win32security.LsaStorePrivateData(policy_handle, keyname, privatedata)
retrieveddata = win32security.LsaRetrievePrivateData(policy_handle, keyname)
assert retrieveddata == privatedata

## passing None deletes key
win32security.LsaStorePrivateData(policy_handle, keyname, None)
win32security.LsaClose(policy_handle)
