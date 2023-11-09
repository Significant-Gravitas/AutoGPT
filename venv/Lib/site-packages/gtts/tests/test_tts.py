# -*- coding: utf-8 -*-
import os
import pytest
from unittest.mock import Mock

from gtts.tts import gTTS, gTTSError
from gtts.langs import _main_langs
from gtts.lang import _extra_langs

# Testing all languages takes some time.
# Set TEST_LANGS envvar to choose languages to test.
#  * 'main': Languages extracted from the Web
#  * 'extra': Languagee set in Languages.EXTRA_LANGS
#  * 'all': All of the above
#  * <csv>: Languages tags list to test
# Unset TEST_LANGS to test everything ('all')
# See: langs_dict()


"""Construct a dict of suites of languages to test.
{ '<suite name>' : <list or dict of language tags> }

ex.: { 'fetch' : {'en': 'English', 'fr': 'French'},
       'extra' : {'en': 'English', 'fr': 'French'} }
ex.: { 'environ' : ['en', 'fr'] }
"""
env = os.environ.get("TEST_LANGS")
if not env or env == "all":
    langs = _main_langs()
    langs.update(_extra_langs())
elif env == "main":
    langs = _main_langs()
elif env == "extra":
    langs = _extra_langs()
else:
    env_langs = {l: l for l in env.split(",") if l}
    langs = env_langs


@pytest.mark.net
@pytest.mark.parametrize("lang", langs.keys(), ids=list(langs.values()))
def test_TTS(tmp_path, lang):
    """Test all supported languages and file save"""

    text = "This is a test"
    """Create output .mp3 file successfully"""
    for slow in (False, True):
        filename = tmp_path / "test_{}_.mp3".format(lang)
        # Create gTTS and save
        tts = gTTS(text=text, lang=lang, slow=slow, lang_check=False)
        tts.save(filename)

        # Check if files created is > 1.5
        assert filename.stat().st_size > 1500


@pytest.mark.net
def test_unsupported_language_check():
    """Raise ValueError on unsupported language (with language check)"""
    lang = "xx"
    text = "Lorem ipsum"
    check = True
    with pytest.raises(ValueError):
        gTTS(text=text, lang=lang, lang_check=check)


def test_empty_string():
    """Raise AssertionError on empty string"""
    text = ""
    with pytest.raises(AssertionError):
        gTTS(text=text)


def test_no_text_parts(tmp_path):
    """Raises AssertionError on no content to send to API (no text_parts)"""
    text = "                                                                                                          ..,\n"
    with pytest.raises(AssertionError):
        filename = tmp_path / "no_content.txt"
        tts = gTTS(text=text)
        tts.save(filename)


# Test write_to_fp()/save() cases not covered elsewhere in this file


@pytest.mark.net
def test_bad_fp_type():
    """Raise TypeError if fp is not a file-like object (no .write())"""
    # Create gTTS and save
    tts = gTTS(text="test")
    with pytest.raises(TypeError):
        tts.write_to_fp(5)


@pytest.mark.net
def test_save(tmp_path):
    """Save .mp3 file successfully"""
    filename = tmp_path / "save.mp3"
    # Create gTTS and save
    tts = gTTS(text="test")
    tts.save(filename)

    # Check if file created is > 2k
    assert filename.stat().st_size > 2000


@pytest.mark.net
def test_get_bodies():
    """get request bodies list"""
    tts = gTTS(text="test", tld="com", lang="en")
    body = tts.get_bodies()[0]
    assert "test" in body
    # \"en\" url-encoded
    assert "%5C%22en%5C%22" in body


def test_msg():
    """Test gTTsError internal exception handling
    Set exception message successfully"""
    error1 = gTTSError("test")
    assert "test" == error1.msg

    error2 = gTTSError()
    assert error2.msg is None


def test_infer_msg():
    """Infer message sucessfully based on context"""

    # Without response:

    # Bad TLD
    ttsTLD = Mock(tld="invalid")
    errorTLD = gTTSError(tts=ttsTLD)
    assert (
        errorTLD.msg
        == "Failed to connect. Probable cause: Host 'https://translate.google.invalid/' is not reachable"
    )

    # With response:

    # 403
    tts403 = Mock()
    response403 = Mock(status_code=403, reason="aaa")
    error403 = gTTSError(tts=tts403, response=response403)
    assert (
        error403.msg
        == "403 (aaa) from TTS API. Probable cause: Bad token or upstream API changes"
    )

    # 200 (and not lang_check)
    tts200 = Mock(lang="xx", lang_check=False)
    response404 = Mock(status_code=200, reason="bbb")
    error200 = gTTSError(tts=tts200, response=response404)
    assert (
        error200.msg
        == "200 (bbb) from TTS API. Probable cause: No audio stream in response. Unsupported language 'xx'"
    )

    # >= 500
    tts500 = Mock()
    response500 = Mock(status_code=500, reason="ccc")
    error500 = gTTSError(tts=tts500, response=response500)
    assert (
        error500.msg
        == "500 (ccc) from TTS API. Probable cause: Uptream API error. Try again later."
    )

    # Unknown (ex. 100)
    tts100 = Mock()
    response100 = Mock(status_code=100, reason="ddd")
    error100 = gTTSError(tts=tts100, response=response100)
    assert error100.msg == "100 (ddd) from TTS API. Probable cause: Unknown"


@pytest.mark.net
def test_WebRequest(tmp_path):
    """Test Web Requests"""

    text = "Lorem ipsum"

    """Raise gTTSError on unsupported language (without language check)"""
    lang = "xx"
    check = False

    with pytest.raises(gTTSError):
        filename = tmp_path / "xx.txt"
        # Create gTTS
        tts = gTTS(text=text, lang=lang, lang_check=check)
        tts.save(filename)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
