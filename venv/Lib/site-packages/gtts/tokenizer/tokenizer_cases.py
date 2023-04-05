# -*- coding: utf-8 -*-
from gtts.tokenizer import RegexBuilder, symbols


def tone_marks():
    """Keep tone-modifying punctuation by matching following character.

    Assumes the `tone_marks` pre-processor was run for cases where there might
    not be any space after a tone-modifying punctuation mark.
    """
    return RegexBuilder(
        pattern_args=symbols.TONE_MARKS, pattern_func=lambda x: u"(?<={}).".format(x)
    ).regex


def period_comma():
    """Period and comma case.

    Match if not preceded by ".<letter>" and only if followed by space.
    Won't cut in the middle/after dotted abbreviations; won't cut numbers.

    Note:
        Won't match if a dotted abbreviation ends a sentence.

    Note:
        Won't match the end of a sentence if not followed by a space.

    """
    return RegexBuilder(
        pattern_args=symbols.PERIOD_COMMA,
        pattern_func=lambda x: r"(?<!\.[a-z]){} ".format(x),
    ).regex


def colon():
    """Colon case.

    Match a colon ":" only if not preceeded by a digit.
    Mainly to prevent a cut in the middle of time notations e.g. 10:01

    """
    return RegexBuilder(
        pattern_args=symbols.COLON, pattern_func=lambda x: r"(?<!\d){}".format(x)
    ).regex


def other_punctuation():
    """Match other punctuation.

    Match other punctuation to split on; punctuation that naturally
    inserts a break in speech.

    """
    punc = "".join(
        set(symbols.ALL_PUNC)
        - set(symbols.TONE_MARKS)
        - set(symbols.PERIOD_COMMA)
        - set(symbols.COLON)
    )
    return RegexBuilder(pattern_args=punc, pattern_func=lambda x: u"{}".format(x)).regex


def legacy_all_punctuation():  # pragma: no cover b/c tested but Coveralls: ¯\_(ツ)_/¯
    """Match all punctuation.

    Use as only tokenizer case to mimic gTTS 1.x tokenization.
    """
    punc = symbols.ALL_PUNC
    return RegexBuilder(pattern_args=punc, pattern_func=lambda x: u"{}".format(x)).regex
