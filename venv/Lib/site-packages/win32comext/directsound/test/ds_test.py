import os
import struct
import sys
import unittest

import pythoncom
import pywintypes
import win32api
import win32com.directsound.directsound as ds
import win32event
from pywin32_testutil import TestSkipped, find_test_fixture

# next two lines are for for debugging:
# import win32com
# import directsound as ds

WAV_FORMAT_PCM = 1
WAV_HEADER_SIZE = struct.calcsize("<4sl4s4slhhllhh4sl")


def wav_header_unpack(data):
    (
        riff,
        riffsize,
        wave,
        fmt,
        fmtsize,
        format,
        nchannels,
        samplespersecond,
        datarate,
        blockalign,
        bitspersample,
        data,
        datalength,
    ) = struct.unpack("<4sl4s4slhhllhh4sl", data)

    if riff != b"RIFF":
        raise ValueError("invalid wav header")

    if fmtsize != 16 or fmt != b"fmt " or data != b"data":
        # fmt chuck is not first chunk, directly followed by data chuck
        # It is nowhere required that they are, it is just very common
        raise ValueError("cannot understand wav header")

    wfx = pywintypes.WAVEFORMATEX()
    wfx.wFormatTag = format
    wfx.nChannels = nchannels
    wfx.nSamplesPerSec = samplespersecond
    wfx.nAvgBytesPerSec = datarate
    wfx.nBlockAlign = blockalign
    wfx.wBitsPerSample = bitspersample

    return wfx, datalength


def wav_header_pack(wfx, datasize):
    return struct.pack(
        "<4sl4s4slhhllhh4sl",
        b"RIFF",
        36 + datasize,
        b"WAVE",
        b"fmt ",
        16,
        wfx.wFormatTag,
        wfx.nChannels,
        wfx.nSamplesPerSec,
        wfx.nAvgBytesPerSec,
        wfx.nBlockAlign,
        wfx.wBitsPerSample,
        b"data",
        datasize,
    )


class WAVEFORMATTest(unittest.TestCase):
    def test_1_Type(self):
        "WAVEFORMATEX type"
        w = pywintypes.WAVEFORMATEX()
        self.assertTrue(type(w) == pywintypes.WAVEFORMATEXType)

    def test_2_Attr(self):
        "WAVEFORMATEX attribute access"
        # A wav header for a soundfile from a CD should look like this...
        w = pywintypes.WAVEFORMATEX()
        w.wFormatTag = pywintypes.WAVE_FORMAT_PCM
        w.nChannels = 2
        w.nSamplesPerSec = 44100
        w.nAvgBytesPerSec = 176400
        w.nBlockAlign = 4
        w.wBitsPerSample = 16

        self.assertTrue(w.wFormatTag == 1)
        self.assertTrue(w.nChannels == 2)
        self.assertTrue(w.nSamplesPerSec == 44100)
        self.assertTrue(w.nAvgBytesPerSec == 176400)
        self.assertTrue(w.nBlockAlign == 4)
        self.assertTrue(w.wBitsPerSample == 16)


class DSCAPSTest(unittest.TestCase):
    def test_1_Type(self):
        "DSCAPS type"
        c = ds.DSCAPS()
        self.assertTrue(type(c) == ds.DSCAPSType)

    def test_2_Attr(self):
        "DSCAPS attribute access"
        c = ds.DSCAPS()
        c.dwFlags = 1
        c.dwMinSecondarySampleRate = 2
        c.dwMaxSecondarySampleRate = 3
        c.dwPrimaryBuffers = 4
        c.dwMaxHwMixingAllBuffers = 5
        c.dwMaxHwMixingStaticBuffers = 6
        c.dwMaxHwMixingStreamingBuffers = 7
        c.dwFreeHwMixingAllBuffers = 8
        c.dwFreeHwMixingStaticBuffers = 9
        c.dwFreeHwMixingStreamingBuffers = 10
        c.dwMaxHw3DAllBuffers = 11
        c.dwMaxHw3DStaticBuffers = 12
        c.dwMaxHw3DStreamingBuffers = 13
        c.dwFreeHw3DAllBuffers = 14
        c.dwFreeHw3DStaticBuffers = 15
        c.dwFreeHw3DStreamingBuffers = 16
        c.dwTotalHwMemBytes = 17
        c.dwFreeHwMemBytes = 18
        c.dwMaxContigFreeHwMemBytes = 19
        c.dwUnlockTransferRateHwBuffers = 20
        c.dwPlayCpuOverheadSwBuffers = 21

        self.assertTrue(c.dwFlags == 1)
        self.assertTrue(c.dwMinSecondarySampleRate == 2)
        self.assertTrue(c.dwMaxSecondarySampleRate == 3)
        self.assertTrue(c.dwPrimaryBuffers == 4)
        self.assertTrue(c.dwMaxHwMixingAllBuffers == 5)
        self.assertTrue(c.dwMaxHwMixingStaticBuffers == 6)
        self.assertTrue(c.dwMaxHwMixingStreamingBuffers == 7)
        self.assertTrue(c.dwFreeHwMixingAllBuffers == 8)
        self.assertTrue(c.dwFreeHwMixingStaticBuffers == 9)
        self.assertTrue(c.dwFreeHwMixingStreamingBuffers == 10)
        self.assertTrue(c.dwMaxHw3DAllBuffers == 11)
        self.assertTrue(c.dwMaxHw3DStaticBuffers == 12)
        self.assertTrue(c.dwMaxHw3DStreamingBuffers == 13)
        self.assertTrue(c.dwFreeHw3DAllBuffers == 14)
        self.assertTrue(c.dwFreeHw3DStaticBuffers == 15)
        self.assertTrue(c.dwFreeHw3DStreamingBuffers == 16)
        self.assertTrue(c.dwTotalHwMemBytes == 17)
        self.assertTrue(c.dwFreeHwMemBytes == 18)
        self.assertTrue(c.dwMaxContigFreeHwMemBytes == 19)
        self.assertTrue(c.dwUnlockTransferRateHwBuffers == 20)
        self.assertTrue(c.dwPlayCpuOverheadSwBuffers == 21)


class DSBCAPSTest(unittest.TestCase):
    def test_1_Type(self):
        "DSBCAPS type"
        c = ds.DSBCAPS()
        self.assertTrue(type(c) == ds.DSBCAPSType)

    def test_2_Attr(self):
        "DSBCAPS attribute access"
        c = ds.DSBCAPS()
        c.dwFlags = 1
        c.dwBufferBytes = 2
        c.dwUnlockTransferRate = 3
        c.dwPlayCpuOverhead = 4

        self.assertTrue(c.dwFlags == 1)
        self.assertTrue(c.dwBufferBytes == 2)
        self.assertTrue(c.dwUnlockTransferRate == 3)
        self.assertTrue(c.dwPlayCpuOverhead == 4)


class DSCCAPSTest(unittest.TestCase):
    def test_1_Type(self):
        "DSCCAPS type"
        c = ds.DSCCAPS()
        self.assertTrue(type(c) == ds.DSCCAPSType)

    def test_2_Attr(self):
        "DSCCAPS attribute access"
        c = ds.DSCCAPS()
        c.dwFlags = 1
        c.dwFormats = 2
        c.dwChannels = 4

        self.assertTrue(c.dwFlags == 1)
        self.assertTrue(c.dwFormats == 2)
        self.assertTrue(c.dwChannels == 4)


class DSCBCAPSTest(unittest.TestCase):
    def test_1_Type(self):
        "DSCBCAPS type"
        c = ds.DSCBCAPS()
        self.assertTrue(type(c) == ds.DSCBCAPSType)

    def test_2_Attr(self):
        "DSCBCAPS attribute access"
        c = ds.DSCBCAPS()
        c.dwFlags = 1
        c.dwBufferBytes = 2

        self.assertTrue(c.dwFlags == 1)
        self.assertTrue(c.dwBufferBytes == 2)


class DSBUFFERDESCTest(unittest.TestCase):
    def test_1_Type(self):
        "DSBUFFERDESC type"
        c = ds.DSBUFFERDESC()
        self.assertTrue(type(c) == ds.DSBUFFERDESCType)

    def test_2_Attr(self):
        "DSBUFFERDESC attribute access"
        c = ds.DSBUFFERDESC()
        c.dwFlags = 1
        c.dwBufferBytes = 2
        c.lpwfxFormat = pywintypes.WAVEFORMATEX()
        c.lpwfxFormat.wFormatTag = pywintypes.WAVE_FORMAT_PCM
        c.lpwfxFormat.nChannels = 2
        c.lpwfxFormat.nSamplesPerSec = 44100
        c.lpwfxFormat.nAvgBytesPerSec = 176400
        c.lpwfxFormat.nBlockAlign = 4
        c.lpwfxFormat.wBitsPerSample = 16

        self.assertTrue(c.dwFlags == 1)
        self.assertTrue(c.dwBufferBytes == 2)
        self.assertTrue(c.lpwfxFormat.wFormatTag == 1)
        self.assertTrue(c.lpwfxFormat.nChannels == 2)
        self.assertTrue(c.lpwfxFormat.nSamplesPerSec == 44100)
        self.assertTrue(c.lpwfxFormat.nAvgBytesPerSec == 176400)
        self.assertTrue(c.lpwfxFormat.nBlockAlign == 4)
        self.assertTrue(c.lpwfxFormat.wBitsPerSample == 16)

    def invalid_format(self, c):
        c.lpwfxFormat = 17

    def test_3_invalid_format(self):
        "DSBUFFERDESC invalid lpwfxFormat assignment"
        c = ds.DSBUFFERDESC()
        self.assertRaises(ValueError, self.invalid_format, c)


class DSCBUFFERDESCTest(unittest.TestCase):
    def test_1_Type(self):
        "DSCBUFFERDESC type"
        c = ds.DSCBUFFERDESC()
        self.assertTrue(type(c) == ds.DSCBUFFERDESCType)

    def test_2_Attr(self):
        "DSCBUFFERDESC attribute access"
        c = ds.DSCBUFFERDESC()
        c.dwFlags = 1
        c.dwBufferBytes = 2
        c.lpwfxFormat = pywintypes.WAVEFORMATEX()
        c.lpwfxFormat.wFormatTag = pywintypes.WAVE_FORMAT_PCM
        c.lpwfxFormat.nChannels = 2
        c.lpwfxFormat.nSamplesPerSec = 44100
        c.lpwfxFormat.nAvgBytesPerSec = 176400
        c.lpwfxFormat.nBlockAlign = 4
        c.lpwfxFormat.wBitsPerSample = 16

        self.assertTrue(c.dwFlags == 1)
        self.assertTrue(c.dwBufferBytes == 2)
        self.assertTrue(c.lpwfxFormat.wFormatTag == 1)
        self.assertTrue(c.lpwfxFormat.nChannels == 2)
        self.assertTrue(c.lpwfxFormat.nSamplesPerSec == 44100)
        self.assertTrue(c.lpwfxFormat.nAvgBytesPerSec == 176400)
        self.assertTrue(c.lpwfxFormat.nBlockAlign == 4)
        self.assertTrue(c.lpwfxFormat.wBitsPerSample == 16)

    def invalid_format(self, c):
        c.lpwfxFormat = 17

    def test_3_invalid_format(self):
        "DSCBUFFERDESC invalid lpwfxFormat assignment"
        c = ds.DSCBUFFERDESC()
        self.assertRaises(ValueError, self.invalid_format, c)


class DirectSoundTest(unittest.TestCase):
    # basic tests - mostly just exercise the functions
    def testEnumerate(self):
        """DirectSoundEnumerate() sanity tests"""

        devices = ds.DirectSoundEnumerate()
        # this might fail on machines without a sound card
        self.assertTrue(len(devices))
        # if we have an entry, it must be a tuple of size 3
        self.assertTrue(len(devices[0]) == 3)

    def testCreate(self):
        """DirectSoundCreate()"""
        try:
            d = ds.DirectSoundCreate(None, None)
        except pythoncom.com_error as exc:
            if exc.hresult != ds.DSERR_NODRIVER:
                raise
            raise TestSkipped(exc)

    def testPlay(self):
        """Mesdames et Messieurs, la cour de Devin Dazzle"""
        # relative to 'testall.py' in the win32com test suite.
        extra = os.path.join(
            os.path.dirname(sys.argv[0]), "../../win32comext/directsound/test"
        )

        fname = find_test_fixture("01-Intro.wav", extra)

        with open(fname, "rb") as f:
            hdr = f.read(WAV_HEADER_SIZE)
            wfx, size = wav_header_unpack(hdr)

            try:
                d = ds.DirectSoundCreate(None, None)
            except pythoncom.com_error as exc:
                if exc.hresult != ds.DSERR_NODRIVER:
                    raise
                raise TestSkipped(exc)
            d.SetCooperativeLevel(None, ds.DSSCL_PRIORITY)

            sdesc = ds.DSBUFFERDESC()
            sdesc.dwFlags = ds.DSBCAPS_STICKYFOCUS | ds.DSBCAPS_CTRLPOSITIONNOTIFY
            sdesc.dwBufferBytes = size
            sdesc.lpwfxFormat = wfx

            buffer = d.CreateSoundBuffer(sdesc, None)

            event = win32event.CreateEvent(None, 0, 0, None)
            notify = buffer.QueryInterface(ds.IID_IDirectSoundNotify)

            notify.SetNotificationPositions((ds.DSBPN_OFFSETSTOP, event))

            buffer.Update(0, f.read(size))

            buffer.Play(0)

            win32event.WaitForSingleObject(event, -1)


class DirectSoundCaptureTest(unittest.TestCase):
    # basic tests - mostly just exercise the functions
    def testEnumerate(self):
        """DirectSoundCaptureEnumerate() sanity tests"""

        devices = ds.DirectSoundCaptureEnumerate()
        # this might fail on machines without a sound card
        self.assertTrue(len(devices))
        # if we have an entry, it must be a tuple of size 3
        self.assertTrue(len(devices[0]) == 3)

    def testCreate(self):
        """DirectSoundCreate()"""
        try:
            d = ds.DirectSoundCaptureCreate(None, None)
        except pythoncom.com_error as exc:
            if exc.hresult != ds.DSERR_NODRIVER:
                raise
            raise TestSkipped(exc)

    def testRecord(self):
        try:
            d = ds.DirectSoundCaptureCreate(None, None)
        except pythoncom.com_error as exc:
            if exc.hresult != ds.DSERR_NODRIVER:
                raise
            raise TestSkipped(exc)

        sdesc = ds.DSCBUFFERDESC()
        sdesc.dwBufferBytes = 352800  # 2 seconds
        sdesc.lpwfxFormat = pywintypes.WAVEFORMATEX()
        sdesc.lpwfxFormat.wFormatTag = pywintypes.WAVE_FORMAT_PCM
        sdesc.lpwfxFormat.nChannels = 2
        sdesc.lpwfxFormat.nSamplesPerSec = 44100
        sdesc.lpwfxFormat.nAvgBytesPerSec = 176400
        sdesc.lpwfxFormat.nBlockAlign = 4
        sdesc.lpwfxFormat.wBitsPerSample = 16

        buffer = d.CreateCaptureBuffer(sdesc)

        event = win32event.CreateEvent(None, 0, 0, None)
        notify = buffer.QueryInterface(ds.IID_IDirectSoundNotify)

        notify.SetNotificationPositions((ds.DSBPN_OFFSETSTOP, event))

        buffer.Start(0)

        win32event.WaitForSingleObject(event, -1)
        event.Close()

        data = buffer.Update(0, 352800)
        fname = os.path.join(win32api.GetTempPath(), "test_directsound_record.wav")
        f = open(fname, "wb")
        f.write(wav_header_pack(sdesc.lpwfxFormat, 352800))
        f.write(data)
        f.close()


if __name__ == "__main__":
    unittest.main()
