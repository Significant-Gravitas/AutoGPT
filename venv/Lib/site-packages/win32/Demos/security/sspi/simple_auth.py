# A demo of basic SSPI authentication.
# There is a 'client' context and a 'server' context - typically these will
# be on different machines (here they are in the same process, but the same
# concepts apply)
import sspi
import sspicon
import win32api
import win32security


def lookup_ret_code(err):
    for k, v in list(sspicon.__dict__.items()):
        if k[0:6] in ("SEC_I_", "SEC_E_") and v == err:
            return k


"""
pkg_name='Kerberos'
sspiclient=SSPIClient(pkg_name, win32api.GetUserName(),  ## target spn is ourself
    None, None,   ## use none for client name and authentication information for current context
    ## u'username', (u'username',u'domain.com',u'passwd'),
    sspicon.ISC_REQ_INTEGRITY|sspicon.ISC_REQ_SEQUENCE_DETECT|sspicon.ISC_REQ_REPLAY_DETECT|    \
        sspicon.ISC_REQ_DELEGATE|sspicon.ISC_REQ_CONFIDENTIALITY|sspicon.ISC_REQ_USE_SESSION_KEY)
sspiserver=SSPIServer(pkg_name, None,
    sspicon.ASC_REQ_INTEGRITY|sspicon.ASC_REQ_SEQUENCE_DETECT|sspicon.ASC_REQ_REPLAY_DETECT|   \
        sspicon.ASC_REQ_DELEGATE|sspicon.ASC_REQ_CONFIDENTIALITY|sspicon.ASC_REQ_STREAM|sspicon.ASC_REQ_USE_SESSION_KEY)
"""

pkg_name = "NTLM"

# Setup the 2 contexts.
sspiclient = sspi.ClientAuth(pkg_name)
sspiserver = sspi.ServerAuth(pkg_name)

# Perform the authentication dance, each loop exchanging more information
# on the way to completing authentication.
sec_buffer = None
while 1:
    err, sec_buffer = sspiclient.authorize(sec_buffer)
    err, sec_buffer = sspiserver.authorize(sec_buffer)
    if err == 0:
        break

# The server can now impersonate the client.  In this demo the 2 users will
# always be the same.
sspiserver.ctxt.ImpersonateSecurityContext()
print("Impersonated user: ", win32api.GetUserNameEx(win32api.NameSamCompatible))
sspiserver.ctxt.RevertSecurityContext()
print("Reverted to self: ", win32api.GetUserName())

pkg_size_info = sspiclient.ctxt.QueryContextAttributes(sspicon.SECPKG_ATTR_SIZES)
# Now sign some data
msg = "some data to be encrypted ......"

sigsize = pkg_size_info["MaxSignature"]
sigbuf = win32security.PySecBufferDescType()
sigbuf.append(win32security.PySecBufferType(len(msg), sspicon.SECBUFFER_DATA))
sigbuf.append(win32security.PySecBufferType(sigsize, sspicon.SECBUFFER_TOKEN))
sigbuf[0].Buffer = msg
sspiclient.ctxt.MakeSignature(0, sigbuf, 1)
sspiserver.ctxt.VerifySignature(sigbuf, 1)

# And finally encrypt some.
trailersize = pkg_size_info["SecurityTrailer"]
encbuf = win32security.PySecBufferDescType()
encbuf.append(win32security.PySecBufferType(len(msg), sspicon.SECBUFFER_DATA))
encbuf.append(win32security.PySecBufferType(trailersize, sspicon.SECBUFFER_TOKEN))
encbuf[0].Buffer = msg
sspiclient.ctxt.EncryptMessage(0, encbuf, 1)
print("Encrypted data:", repr(encbuf[0].Buffer))
sspiserver.ctxt.DecryptMessage(encbuf, 1)
print("Unencrypted data:", encbuf[0].Buffer)
