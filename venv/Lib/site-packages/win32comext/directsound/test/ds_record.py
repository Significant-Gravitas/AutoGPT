import os
import struct

import pywintypes
import win32api
import win32com.directsound.directsound as ds
import win32event


def wav_header_pack(wfx, datasize):
    return struct.pack(
        "<4sl4s4slhhllhh4sl",
        "RIFF",
        36 + datasize,
        "WAVE",
        "fmt ",
        16,
        wfx.wFormatTag,
        wfx.nChannels,
        wfx.nSamplesPerSec,
        wfx.nAvgBytesPerSec,
        wfx.nBlockAlign,
        wfx.wBitsPerSample,
        "data",
        datasize,
    )


d = ds.DirectSoundCaptureCreate(None, None)

sdesc = ds.DSCBUFFERDESC()
sdesc.dwBufferBytes = 352800  # 2 seconds
sdesc.lpwfxFormat = pywintypes.WAVEFORMATEX()
sdesc.lpwfxFormat.wFormatTag = pywintypes.WAVE_FORMAT_PCM
sdesc.lpwfxFormat.nChannels = 2
sdesc.lpwfxFormat.nSamplesPerSec = 44100
sdesc.lpwfxFormat.nAvgBytesPerSec = 176400
sdesc.lpwfxFormat.nBlockAlign = 4
sdesc.lpwfxFormat.wBitsPerSample = 16

print(sdesc)
print(d)
buffer = d.CreateCaptureBuffer(sdesc)

event = win32event.CreateEvent(None, 0, 0, None)
notify = buffer.QueryInterface(ds.IID_IDirectSoundNotify)

notify.SetNotificationPositions((ds.DSBPN_OFFSETSTOP, event))

buffer.Start(0)

win32event.WaitForSingleObject(event, -1)

data = buffer.Update(0, 352800)

fname = os.path.join(win32api.GetTempPath(), "test_directsound_record.wav")
f = open(fname, "wb")
f.write(wav_header_pack(sdesc.lpwfxFormat, 352800))
f.write(data)
f.close()
