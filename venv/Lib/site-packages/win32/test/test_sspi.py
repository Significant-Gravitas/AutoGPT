# Some tests of the win32security sspi functions.
# Stolen from Roger's original test_sspi.c, a version of which is in "Demos"
# See also the other SSPI demos.
import re
import unittest

import sspi
import sspicon
import win32api
import win32security
from pywin32_testutil import TestSkipped, str2bytes, testmain


# It is quite likely that the Kerberos tests will fail due to not being
# installed.  The NTLM tests do *not* get the same behaviour as they should
# always be there.
def applyHandlingSkips(func, *args):
    try:
        return func(*args)
    except win32api.error as exc:
        if exc.winerror in [
            sspicon.SEC_E_NO_CREDENTIALS,
            sspicon.SEC_E_NO_AUTHENTICATING_AUTHORITY,
        ]:
            raise TestSkipped(exc)
        raise


class TestSSPI(unittest.TestCase):
    def assertRaisesHRESULT(self, hr, func, *args):
        try:
            return func(*args)
            raise RuntimeError("expecting %s failure" % (hr,))
        except win32security.error as exc:
            self.assertEqual(exc.winerror, hr)

    def _doAuth(self, pkg_name):
        sspiclient = sspi.ClientAuth(pkg_name, targetspn=win32api.GetUserName())
        sspiserver = sspi.ServerAuth(pkg_name)

        sec_buffer = None
        err = 1
        while err != 0:
            err, sec_buffer = sspiclient.authorize(sec_buffer)
            err, sec_buffer = sspiserver.authorize(sec_buffer)
        return sspiclient, sspiserver

    def _doTestImpersonate(self, pkg_name):
        # Just for the sake of code exercising!
        sspiclient, sspiserver = self._doAuth(pkg_name)
        sspiserver.ctxt.ImpersonateSecurityContext()
        sspiserver.ctxt.RevertSecurityContext()

    def testImpersonateKerberos(self):
        applyHandlingSkips(self._doTestImpersonate, "Kerberos")

    def testImpersonateNTLM(self):
        self._doTestImpersonate("NTLM")

    def _doTestEncrypt(self, pkg_name):
        sspiclient, sspiserver = self._doAuth(pkg_name)

        pkg_size_info = sspiclient.ctxt.QueryContextAttributes(
            sspicon.SECPKG_ATTR_SIZES
        )
        msg = str2bytes("some data to be encrypted ......")

        trailersize = pkg_size_info["SecurityTrailer"]
        encbuf = win32security.PySecBufferDescType()
        encbuf.append(win32security.PySecBufferType(len(msg), sspicon.SECBUFFER_DATA))
        encbuf.append(
            win32security.PySecBufferType(trailersize, sspicon.SECBUFFER_TOKEN)
        )
        encbuf[0].Buffer = msg
        sspiclient.ctxt.EncryptMessage(0, encbuf, 1)
        sspiserver.ctxt.DecryptMessage(encbuf, 1)
        self.assertEqual(msg, encbuf[0].Buffer)
        # and test the higher-level functions
        data_in = str2bytes("hello")
        data, sig = sspiclient.encrypt(data_in)
        self.assertEqual(sspiserver.decrypt(data, sig), data_in)

        data, sig = sspiserver.encrypt(data_in)
        self.assertEqual(sspiclient.decrypt(data, sig), data_in)

    def _doTestEncryptStream(self, pkg_name):
        # Test out the SSPI/GSSAPI interop wrapping examples at
        # https://docs.microsoft.com/en-us/windows/win32/secauthn/sspi-kerberos-interoperability-with-gssapi

        sspiclient, sspiserver = self._doAuth(pkg_name)

        pkg_size_info = sspiclient.ctxt.QueryContextAttributes(
            sspicon.SECPKG_ATTR_SIZES
        )
        msg = str2bytes("some data to be encrypted ......")

        trailersize = pkg_size_info["SecurityTrailer"]
        blocksize = pkg_size_info["BlockSize"]
        encbuf = win32security.PySecBufferDescType()
        encbuf.append(
            win32security.PySecBufferType(trailersize, sspicon.SECBUFFER_TOKEN)
        )
        encbuf.append(win32security.PySecBufferType(len(msg), sspicon.SECBUFFER_DATA))
        encbuf.append(
            win32security.PySecBufferType(blocksize, sspicon.SECBUFFER_PADDING)
        )
        encbuf[1].Buffer = msg
        sspiclient.ctxt.EncryptMessage(0, encbuf, 1)

        encmsg = encbuf[0].Buffer + encbuf[1].Buffer + encbuf[2].Buffer
        decbuf = win32security.PySecBufferDescType()
        decbuf.append(
            win32security.PySecBufferType(len(encmsg), sspicon.SECBUFFER_STREAM)
        )
        decbuf.append(win32security.PySecBufferType(0, sspicon.SECBUFFER_DATA))
        decbuf[0].Buffer = encmsg

        sspiserver.ctxt.DecryptMessage(decbuf, 1)
        self.assertEqual(msg, decbuf[1].Buffer)

    def testEncryptNTLM(self):
        self._doTestEncrypt("NTLM")

    def testEncryptStreamNTLM(self):
        self._doTestEncryptStream("NTLM")

    def testEncryptKerberos(self):
        applyHandlingSkips(self._doTestEncrypt, "Kerberos")

    def testEncryptStreamKerberos(self):
        applyHandlingSkips(self._doTestEncryptStream, "Kerberos")

    def _doTestSign(self, pkg_name):
        sspiclient, sspiserver = self._doAuth(pkg_name)

        pkg_size_info = sspiclient.ctxt.QueryContextAttributes(
            sspicon.SECPKG_ATTR_SIZES
        )
        msg = str2bytes("some data to be encrypted ......")

        sigsize = pkg_size_info["MaxSignature"]
        sigbuf = win32security.PySecBufferDescType()
        sigbuf.append(win32security.PySecBufferType(len(msg), sspicon.SECBUFFER_DATA))
        sigbuf.append(win32security.PySecBufferType(sigsize, sspicon.SECBUFFER_TOKEN))
        sigbuf[0].Buffer = msg
        sspiclient.ctxt.MakeSignature(0, sigbuf, 0)
        sspiserver.ctxt.VerifySignature(sigbuf, 0)
        # and test the higher-level functions
        sspiclient.next_seq_num = 1
        sspiserver.next_seq_num = 1
        data = str2bytes("hello")
        key = sspiclient.sign(data)
        sspiserver.verify(data, key)
        key = sspiclient.sign(data)
        self.assertRaisesHRESULT(
            sspicon.SEC_E_MESSAGE_ALTERED, sspiserver.verify, data + data, key
        )

        # and the other way
        key = sspiserver.sign(data)
        sspiclient.verify(data, key)
        key = sspiserver.sign(data)
        self.assertRaisesHRESULT(
            sspicon.SEC_E_MESSAGE_ALTERED, sspiclient.verify, data + data, key
        )

    def testSignNTLM(self):
        self._doTestSign("NTLM")

    def testSignKerberos(self):
        applyHandlingSkips(self._doTestSign, "Kerberos")

    def _testSequenceSign(self):
        # Only Kerberos supports sequence detection.
        sspiclient, sspiserver = self._doAuth("Kerberos")
        key = sspiclient.sign(b"hello")
        sspiclient.sign(b"hello")
        self.assertRaisesHRESULT(
            sspicon.SEC_E_OUT_OF_SEQUENCE, sspiserver.verify, b"hello", key
        )

    def testSequenceSign(self):
        applyHandlingSkips(self._testSequenceSign)

    def _testSequenceEncrypt(self):
        # Only Kerberos supports sequence detection.
        sspiclient, sspiserver = self._doAuth("Kerberos")
        blob, key = sspiclient.encrypt(b"hello")
        blob, key = sspiclient.encrypt(b"hello")
        self.assertRaisesHRESULT(
            sspicon.SEC_E_OUT_OF_SEQUENCE, sspiserver.decrypt, blob, key
        )

    def testSequenceEncrypt(self):
        applyHandlingSkips(self._testSequenceEncrypt)

    def testSecBufferRepr(self):
        desc = win32security.PySecBufferDescType()
        assert re.match(
            "PySecBufferDesc\(ulVersion: 0 \| cBuffers: 0 \| pBuffers: 0x[\da-fA-F]{8,16}\)",
            repr(desc),
        )

        buffer1 = win32security.PySecBufferType(0, sspicon.SECBUFFER_TOKEN)
        assert re.match(
            "PySecBuffer\(cbBuffer: 0 \| BufferType: 2 \| pvBuffer: 0x[\da-fA-F]{8,16}\)",
            repr(buffer1),
        )
        "PySecBuffer(cbBuffer: 0 | BufferType: 2 | pvBuffer: 0x000001B8CC6D8020)"
        desc.append(buffer1)

        assert re.match(
            "PySecBufferDesc\(ulVersion: 0 \| cBuffers: 1 \| pBuffers: 0x[\da-fA-F]{8,16}\)",
            repr(desc),
        )

        buffer2 = win32security.PySecBufferType(4, sspicon.SECBUFFER_DATA)
        assert re.match(
            "PySecBuffer\(cbBuffer: 4 \| BufferType: 1 \| pvBuffer: 0x[\da-fA-F]{8,16}\)",
            repr(buffer2),
        )
        desc.append(buffer2)

        assert re.match(
            "PySecBufferDesc\(ulVersion: 0 \| cBuffers: 2 \| pBuffers: 0x[\da-fA-F]{8,16}\)",
            repr(desc),
        )


if __name__ == "__main__":
    testmain()
