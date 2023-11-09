# -*- coding: utf-8 -*-
import pytest
from gtts.utils import _minimize, _len, _clean_tokens, _translate_url

delim = " "
Lmax = 10


def test_ascii():
    _in = "Bacon ipsum dolor sit amet"
    _out = ["Bacon", "ipsum", "dolor sit", "amet"]
    assert _minimize(_in, delim, Lmax) == _out


def test_ascii_no_delim():
    _in = "Baconipsumdolorsitametflankcornedbee"
    _out = ["Baconipsum", "dolorsitam", "etflankcor", "nedbee"]
    assert _minimize(_in, delim, Lmax) == _out


def test_unicode():
    _in = u"这是一个三岁的小孩在讲述他从一系列照片里看到的东西。"
    _out = [u"这是一个三岁的小孩在", u"讲述他从一系列照片里", u"看到的东西。"]
    assert _minimize(_in, delim, Lmax) == _out


def test_startwith_delim():
    _in = delim + "test"
    _out = ["test"]
    assert _minimize(_in, delim, Lmax) == _out


def test_len_ascii():
    text = "Bacon ipsum dolor sit amet flank corned beef."
    assert _len(text) == 45


def test_len_unicode():
    text = u"但在一个重要的任务上"
    assert _len(text) == 10


def test_only_space_and_punc():
    _in = [",(:)?", "\t    ", "\n"]
    _out = []
    assert _clean_tokens(_in) == _out


def test_strip():
    _in = [" Bacon  ", "& ", "ipsum\r", "."]
    _out = ["Bacon", "&", "ipsum"]
    assert _clean_tokens(_in) == _out


def test_translate_url():
    _in = {"tld": "qwerty", "path": "asdf"}
    _out = "https://translate.google.qwerty/asdf"
    assert _translate_url(**_in) == _out


if __name__ == "__main__":
    pytest.main(["-x", __file__])
