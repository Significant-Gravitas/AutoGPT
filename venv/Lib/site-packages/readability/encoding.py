import re
import chardet
import sys


RE_CHARSET = re.compile(br'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
RE_PRAGMA = re.compile(br'<meta.*?content=["\']*;?charset=(.+?)["\'>]', flags=re.I)
RE_XML = re.compile(br'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

CHARSETS = {
    "big5": "big5hkscs",
    "gb2312": "gb18030",
    "ascii": "utf-8",
    "maccyrillic": "cp1251",
    "win1251": "cp1251",
    "win-1251": "cp1251",
    "windows-1251": "cp1251",
}


def fix_charset(encoding):
    """Overrides encoding when charset declaration
       or charset determination is a subset of a larger
       charset.  Created because of issues with Chinese websites"""
    encoding = encoding.lower()
    return CHARSETS.get(encoding, encoding)


def get_encoding(page):
    # Regex for XML and HTML Meta charset declaration
    declared_encodings = (
        RE_CHARSET.findall(page) + RE_PRAGMA.findall(page) + RE_XML.findall(page)
    )

    # Try any declared encodings
    for declared_encoding in declared_encodings:
        try:
            if sys.version_info[0] == 3:
                # declared_encoding will actually be bytes but .decode() only
                # accepts `str` type. Decode blindly with ascii because no one should
                # ever use non-ascii characters in the name of an encoding.
                declared_encoding = declared_encoding.decode("ascii", "replace")

            encoding = fix_charset(declared_encoding)

            # Now let's decode the page
            page.decode(encoding)
            # It worked!
            return encoding
        except (UnicodeDecodeError, LookupError):
            pass

    # Fallback to chardet if declared encodings fail
    # Remove all HTML tags, and leave only text for chardet
    text = re.sub(br"(\s*</?[^>]*>)+\s*", b" ", page).strip()
    enc = "utf-8"
    if len(text) < 10:
        return enc  # can't guess
    res = chardet.detect(text)
    enc = res["encoding"] or "utf-8"
    # print '->', enc, "%.2f" % res['confidence']
    enc = fix_charset(enc)
    return enc
