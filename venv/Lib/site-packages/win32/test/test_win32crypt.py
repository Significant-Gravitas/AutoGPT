# Test module for win32crypt

import contextlib
import unittest
from typing import Any, Iterator

import win32crypt
from pywin32_testutil import TestSkipped, find_test_fixture, testmain
from win32cryptcon import *


class Crypt(unittest.TestCase):
    def testSimple(self):
        data = b"My test data"
        entropy = None
        desc = "My description"
        flags = 0
        ps = None
        blob = win32crypt.CryptProtectData(data, desc, entropy, None, ps, flags)
        got_desc, got_data = win32crypt.CryptUnprotectData(
            blob, entropy, None, ps, flags
        )
        self.assertEqual(data, got_data)
        self.assertEqual(desc, got_desc)

    def testEntropy(self):
        data = b"My test data"
        entropy = b"My test entropy"
        desc = "My description"
        flags = 0
        ps = None
        blob = win32crypt.CryptProtectData(data, desc, entropy, None, ps, flags)
        got_desc, got_data = win32crypt.CryptUnprotectData(
            blob, entropy, None, ps, flags
        )
        self.assertEqual(data, got_data)
        self.assertEqual(desc, got_desc)


# via https://github.com/mhammond/pywin32/issues/1859
_LOCAL_MACHINE = "LocalMachine"
_CURRENT_USER = "CurrentUser"


@contextlib.contextmanager
def open_windows_certstore(store_name: str, store_location: str) -> Iterator[Any]:
    """Open a windows certificate store

    :param store_name: store name
    :param store_location: store location
    :return: handle to cert store
    """
    handle = None
    try:
        handle = win32crypt.CertOpenStore(
            CERT_STORE_PROV_SYSTEM,
            0,
            None,
            CERT_SYSTEM_STORE_LOCAL_MACHINE
            if store_location == _LOCAL_MACHINE
            else CERT_SYSTEM_STORE_CURRENT_USER,
            store_name,
        )
        yield handle
    finally:
        if handle is not None:
            handle.CertCloseStore()


class TestCerts(unittest.TestCase):
    def readCertFile(self, file_name):
        with open(find_test_fixture(file_name), "rb") as f:
            buf = bytearray(f.read())
            return win32crypt.CryptQueryObject(
                CERT_QUERY_OBJECT_BLOB,
                buf,
                CERT_QUERY_CONTENT_FLAG_CERT,
                CERT_QUERY_FORMAT_FLAG_ALL,
                0,
            )

    def testReadCertFiles(self):
        # readCertFile has Python read the file and load it as a blob.
        # win32crypt can read the file directly - let's check that works too
        # (ideally we'd compare the 2 approaches etc, but the objects don't support
        # equality checks etc, so this will do for now.)
        # No need to do this for different filenames!
        filename = "win32crypt_testcert_base64.cer"
        cert = win32crypt.CryptQueryObject(
            CERT_QUERY_OBJECT_FILE,
            find_test_fixture(filename),
            CERT_QUERY_CONTENT_FLAG_CERT,
            CERT_QUERY_FORMAT_FLAG_ALL,
            0,
        )
        self.assertEqual(cert["FormatType"], CERT_QUERY_FORMAT_BASE64_ENCODED)
        self.assertEqual(cert["ContentType"], CERT_QUERY_CONTENT_CERT)

    def checkCertFile(self, filename, expected_format):
        cert = self.readCertFile(filename)
        self.assertEqual(cert["FormatType"], expected_format)
        self.assertEqual(cert["ContentType"], CERT_QUERY_CONTENT_CERT)

        with open_windows_certstore(_CURRENT_USER, "Temp") as store:
            context = store.CertAddCertificateContextToStore(
                cert["Context"], CERT_STORE_ADD_REPLACE_EXISTING
            )
            # Getting 2 certs here - main thing is we get 1!
            self.assertTrue(len(store.CertEnumCertificatesInStore()))
            self.assertFalse(len(store.CertEnumCTLsInStore()))
            context.CertFreeCertificateContext()
            try:
                context.CertFreeCertificateContext()
            except ValueError:
                pass
            else:
                raise RuntimeError("should not be able to close the context twice")

    def testCertBase64(self):
        self.checkCertFile(
            "win32crypt_testcert_base64.cer", CERT_QUERY_FORMAT_BASE64_ENCODED
        )

    def testCertBinary(self):
        self.checkCertFile("win32crypt_testcert_bin.cer", CERT_QUERY_FORMAT_BINARY)


if __name__ == "__main__":
    testmain()
