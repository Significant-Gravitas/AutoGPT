# -*- coding: utf-8 -*-
import pytest
from gtts.lang import tts_langs, _extra_langs, _fallback_deprecated_lang
from gtts.langs import _main_langs

"""Test language list"""


def test_main_langs():
    """Fetch languages successfully"""
    # Safe to assume 'en' (English) will always be there
    scraped_langs = _main_langs()
    assert "en" in scraped_langs


def test_deprecated_lang():
    """Test language deprecation fallback"""
    with pytest.deprecated_call():
        assert _fallback_deprecated_lang("en-gb") == "en"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
