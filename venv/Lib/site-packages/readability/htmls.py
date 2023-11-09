from lxml.html import tostring
import lxml.html
import re

from .cleaners import normalize_spaces, clean_attributes
from .encoding import get_encoding
from .compat import str_

utf8_parser = lxml.html.HTMLParser(encoding="utf-8")


def build_doc(page):
    if isinstance(page, str_):
        encoding = None
        decoded_page = page
    else:
        encoding = get_encoding(page) or "utf-8"
        decoded_page = page.decode(encoding, "replace")

    # XXX: we have to do .decode and .encode even for utf-8 pages to remove bad characters
    doc = lxml.html.document_fromstring(
        decoded_page.encode("utf-8", "replace"), parser=utf8_parser
    )
    return doc, encoding


def js_re(src, pattern, flags, repl):
    return re.compile(pattern, flags).sub(src, repl.replace("$", "\\"))


def normalize_entities(cur_title):
    entities = {
        u"\u2014": "-",
        u"\u2013": "-",
        u"&mdash;": "-",
        u"&ndash;": "-",
        u"\u00A0": " ",
        u"\u00AB": '"',
        u"\u00BB": '"',
        u"&quot;": '"',
    }
    for c, r in entities.items():
        if c in cur_title:
            cur_title = cur_title.replace(c, r)

    return cur_title


def norm_title(title):
    return normalize_entities(normalize_spaces(title))


def get_title(doc):
    title = doc.find(".//title")
    if title is None or title.text is None or len(title.text) == 0:
        return "[no-title]"

    return norm_title(title.text)


def add_match(collection, text, orig):
    text = norm_title(text)
    if len(text.split()) >= 2 and len(text) >= 15:
        if text.replace('"', "") in orig.replace('"', ""):
            collection.add(text)


TITLE_CSS_HEURISTICS = [
    "#title",
    "#head",
    "#heading",
    ".pageTitle",
    ".news_title",
    ".title",
    ".head",
    ".heading",
    ".contentheading",
    ".small_header_red",
]


def shorten_title(doc):
    title = doc.find(".//title")
    if title is None or title.text is None or len(title.text) == 0:
        return ""

    title = orig = norm_title(title.text)

    candidates = set()

    for item in [".//h1", ".//h2", ".//h3"]:
        for e in list(doc.iterfind(item)):
            if e.text:
                add_match(candidates, e.text, orig)
            if e.text_content():
                add_match(candidates, e.text_content(), orig)

    for item in TITLE_CSS_HEURISTICS:
        for e in doc.cssselect(item):
            if e.text:
                add_match(candidates, e.text, orig)
            if e.text_content():
                add_match(candidates, e.text_content(), orig)

    if candidates:
        title = sorted(candidates, key=len)[-1]
    else:
        for delimiter in [" | ", " - ", " :: ", " / "]:
            if delimiter in title:
                parts = orig.split(delimiter)
                if len(parts[0].split()) >= 4:
                    title = parts[0]
                    break
                elif len(parts[-1].split()) >= 4:
                    title = parts[-1]
                    break
        else:
            if ": " in title:
                parts = orig.split(": ")
                if len(parts[-1].split()) >= 4:
                    title = parts[-1]
                else:
                    title = orig.split(": ", 1)[1]

    if not 15 < len(title) < 150:
        return orig

    return title


# is it necessary? Cleaner from LXML is initialized correctly in cleaners.py
def get_body(doc):
    for elem in doc.xpath(".//script | .//link | .//style"):
        elem.drop_tree()
    # tostring() always return utf-8 encoded string
    # FIXME: isn't better to use tounicode?
    raw_html = str_(tostring(doc.body or doc))
    cleaned = clean_attributes(raw_html)
    try:
        # BeautifulSoup(cleaned) #FIXME do we really need to try loading it?
        return cleaned
    except Exception:  # FIXME find the equivalent lxml error
        # logging.error("cleansing broke html content: %s\n---------\n%s" % (raw_html, cleaned))
        return raw_html
