import numpy as np

from backend.voice_relay import vad
from backend.voice_relay.config import IN_RATE
from backend.voice_relay.conftest import silence, speech


def test_speech_is_heard():
    assert vad.heard(speech(0.04)[0])


def test_silence_is_not_heard():
    assert not vad.heard(silence(0.04)[0])


def test_a_click_is_not_heard():
    frame = np.zeros(int(IN_RATE * 0.04), dtype=np.int16)
    frame[:160] = 20_000
    assert not vad.heard(frame.tobytes())


def test_a_quiet_room_is_not_heard():
    hum = (np.sin(np.arange(int(IN_RATE * 0.04)) * 0.3) * 150).astype(np.int16)
    assert not vad.heard(hum.tobytes())


def test_a_frame_with_a_stray_byte_is_still_read():
    assert vad.heard(speech(0.04)[0] + b"\x01")
    assert not vad.heard(b"\x01")
