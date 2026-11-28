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


def _tone(seconds: float) -> bytes:
    return b"".join(speech(seconds))


def _quiet(seconds: float) -> bytes:
    return b"".join(silence(seconds))


def test_speech_is_cut_into_padded_segments():
    pcm = _quiet(1.0) + _tone(0.6) + _quiet(1.0)
    [(start, end)] = vad.segments(pcm)
    assert start == int(0.8 * IN_RATE)
    assert end == int(1.8 * IN_RATE)


def test_speech_across_a_short_pause_is_one_segment():
    pcm = _tone(0.4) + _quiet(0.4) + _tone(0.4) + _quiet(1.0)
    assert len(vad.segments(pcm)) == 1


def test_speech_across_a_long_pause_is_two_segments():
    pcm = _tone(0.4) + _quiet(1.0) + _tone(0.4)
    assert len(vad.segments(pcm)) == 2


def test_a_segment_never_reaches_past_the_recording():
    pcm = _tone(0.4)
    assert vad.segments(pcm) == [(0, len(pcm) // 2)]


def test_a_click_makes_no_segment():
    assert vad.segments(_quiet(0.5) + _tone(0.04) + _quiet(0.5)) == []
