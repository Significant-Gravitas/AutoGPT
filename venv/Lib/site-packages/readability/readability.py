#!/usr/bin/env python
from __future__ import print_function
import logging
import re
import sys

from lxml.etree import tounicode
from lxml.html import document_fromstring
from lxml.html import fragment_fromstring

from .cleaners import clean_attributes
from .cleaners import html_cleaner
from .htmls import build_doc
from .htmls import get_body
from .htmls import get_title
from .htmls import shorten_title
from .compat import str_, bytes_, tostring_
from .debug import describe, text_content


log = logging.getLogger("readability.readability")

REGEXES = {
    "unlikelyCandidatesRe": re.compile(
        r"combx|comment|community|disqus|extra|foot|header|menu|remark|rss|shoutbox|sidebar|sponsor|ad-break|agegate|pagination|pager|popup|tweet|twitter",
        re.I,
    ),
    "okMaybeItsACandidateRe": re.compile(r"and|article|body|column|main|shadow", re.I),
    "positiveRe": re.compile(
        r"article|body|content|entry|hentry|main|page|pagination|post|text|blog|story",
        re.I,
    ),
    "negativeRe": re.compile(
        r"combx|comment|com-|contact|foot|footer|footnote|masthead|media|meta|outbrain|promo|related|scroll|shoutbox|sidebar|sponsor|shopping|tags|tool|widget",
        re.I,
    ),
    "divToPElementsRe": re.compile(
        r"<(a|blockquote|dl|div|img|ol|p|pre|table|ul)", re.I
    ),
    #'replaceBrsRe': re.compile(r'(<br[^>]*>[ \n\r\t]*){2,}',re.I),
    #'replaceFontsRe': re.compile(r'<(\/?)font[^>]*>',re.I),
    #'trimRe': re.compile(r'^\s+|\s+$/'),
    #'normalizeRe': re.compile(r'\s{2,}/'),
    #'killBreaksRe': re.compile(r'(<br\s*\/?>(\s|&nbsp;?)*){1,}/'),
    "videoRe": re.compile(r"https?:\/\/(www\.)?(youtube|vimeo)\.com", re.I),
    # skipFootnoteLink:      /^\s*(\[?[a-z0-9]{1,2}\]?|^|edit|citation needed)\s*$/i,
}


class Unparseable(ValueError):
    pass


def to_int(x):
    if not x:
        return None
    x = x.strip()
    if x.endswith("px"):
        return int(x[:-2])
    if x.endswith("em"):
        return int(x[:-2]) * 12
    return int(x)


def clean(text):
    # Many spaces make the following regexes run forever
    text = re.sub(r"\s{255,}", " " * 255, text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\t|[ \t]{2,}", " ", text)
    return text.strip()


def text_length(i):
    return len(clean(i.text_content() or ""))


def compile_pattern(elements):
    if not elements:
        return None
    elif isinstance(elements, re._pattern_type):
        return elements
    elif isinstance(elements, (str_, bytes_)):
        if isinstance(elements, bytes_):
            elements = str_(elements, "utf-8")
        elements = elements.split(u",")
    if isinstance(elements, (list, tuple)):
        return re.compile(u"|".join([re.escape(x.strip()) for x in elements]), re.U)
    else:
        raise Exception("Unknown type for the pattern: {}".format(type(elements)))
        # assume string or string like object


class Document:
    """Class to build a etree document out of html."""

    def __init__(
        self,
        input,
        positive_keywords=None,
        negative_keywords=None,
        url=None,
        min_text_length=25,
        retry_length=250,
        xpath=False,
        handle_failures="discard",
    ):
        """Generate the document

        :param input: string of the html content.
        :param positive_keywords: regex, list or comma-separated string of patterns in classes and ids
        :param negative_keywords: regex, list or comma-separated string in classes and ids
        :param min_text_length: Tunable. Set to a higher value for more precise detection of longer texts.
        :param retry_length: Tunable. Set to a lower value for better detection of very small texts.
        :param xpath: If set to True, adds x="..." attribute to each HTML node,
        containing xpath path pointing to original document path (allows to
        reconstruct selected summary in original document).
        :param handle_failures: Parameter passed to `lxml` for handling failure during exception.
        Support options = ["discard", "ignore", None]

        Examples:
            positive_keywords=["news-item", "block"]
            positive_keywords=["news-item, block"]
            positive_keywords=re.compile("news|block")
            negative_keywords=["mysidebar", "related", "ads"]

        The Document class is not re-enterable.
        It is designed to create a new Document() for each HTML file to process it.

        API methods:
        .title() -- full title
        .short_title() -- cleaned up title
        .content() -- full content
        .summary() -- cleaned up content
        """
        self.input = input
        self.html = None
        self.encoding = None
        self.positive_keywords = compile_pattern(positive_keywords)
        self.negative_keywords = compile_pattern(negative_keywords)
        self.url = url
        self.min_text_length = min_text_length
        self.retry_length = retry_length
        self.xpath = xpath
        self.handle_failures = handle_failures

    def _html(self, force=False):
        if force or self.html is None:
            self.html = self._parse(self.input)
            if self.xpath:
                root = self.html.getroottree()
                for i in self.html.getiterator():
                    # print root.getpath(i)
                    i.attrib["x"] = root.getpath(i)
        return self.html

    def _parse(self, input):
        doc, self.encoding = build_doc(input)
        doc = html_cleaner.clean_html(doc)
        base_href = self.url
        if base_href:
            # trying to guard against bad links like <a href="http://[http://...">
            try:
                # such support is added in lxml 3.3.0
                doc.make_links_absolute(
                    base_href,
                    resolve_base_href=True,
                    handle_failures=self.handle_failures,
                )
            except TypeError:  # make_links_absolute() got an unexpected keyword argument 'handle_failures'
                # then we have lxml < 3.3.0
                # please upgrade to lxml >= 3.3.0 if you're failing here!
                doc.make_links_absolute(
                    base_href,
                    resolve_base_href=True,
                    handle_failures=self.handle_failures,
                )
        else:
            doc.resolve_base_href(handle_failures=self.handle_failures)
        return doc

    def content(self):
        """Returns document body"""
        return get_body(self._html(True))

    def title(self):
        """Returns document title"""
        return get_title(self._html(True))

    def short_title(self):
        """Returns cleaned up document title"""
        return shorten_title(self._html(True))

    def get_clean_html(self):
        """
        An internal method, which can be overridden in subclasses, for example,
        to disable or to improve DOM-to-text conversion in .summary() method
        """
        return clean_attributes(tounicode(self.html, method="html"))

    def summary(self, html_partial=False):
        """
        Given a HTML file, extracts the text of the article.

        :param html_partial: return only the div of the document, don't wrap
                             in html and body tags.

        Warning: It mutates internal DOM representation of the HTML document,
        so it is better to call other API methods before this one.
        """
        try:
            ruthless = True
            while True:
                self._html(True)
                for i in self.tags(self.html, "script", "style"):
                    i.drop_tree()
                for i in self.tags(self.html, "body"):
                    i.set("id", "readabilityBody")
                if ruthless:
                    self.remove_unlikely_candidates()
                self.transform_misused_divs_into_paragraphs()
                candidates = self.score_paragraphs()

                best_candidate = self.select_best_candidate(candidates)

                if best_candidate:
                    article = self.get_article(
                        candidates, best_candidate, html_partial=html_partial
                    )
                else:
                    if ruthless:
                        log.info("ruthless removal did not work. ")
                        ruthless = False
                        log.debug(
                            (
                                "ended up stripping too much - "
                                "going for a safer _parse"
                            )
                        )
                        # try again
                        continue
                    else:
                        log.debug(
                            (
                                "Ruthless and lenient parsing did not work. "
                                "Returning raw html"
                            )
                        )
                        article = self.html.find("body")
                        if article is None:
                            article = self.html
                cleaned_article = self.sanitize(article, candidates)

                article_length = len(cleaned_article or "")
                retry_length = self.retry_length
                of_acceptable_length = article_length >= retry_length
                if ruthless and not of_acceptable_length:
                    ruthless = False
                    # Loop through and try again.
                    continue
                else:
                    return cleaned_article
        except Exception as e:
            log.exception("error getting summary: ")
            if sys.version_info[0] == 2:
                from .compat.two import raise_with_traceback
            else:
                from .compat.three import raise_with_traceback
            raise_with_traceback(Unparseable, sys.exc_info()[2], str_(e))

    def get_article(self, candidates, best_candidate, html_partial=False):
        # Now that we have the top candidate, look through its siblings for
        # content that might also be related.
        # Things like preambles, content split by ads that we removed, etc.
        sibling_score_threshold = max([10, best_candidate["content_score"] * 0.2])
        # create a new html document with a html->body->div
        if html_partial:
            output = fragment_fromstring("<div/>")
        else:
            output = document_fromstring("<div/>")
        best_elem = best_candidate["elem"]
        parent = best_elem.getparent()
        siblings = parent.getchildren() if parent is not None else [best_elem]
        for sibling in siblings:
            # in lxml there no concept of simple text
            # if isinstance(sibling, NavigableString): continue
            append = False
            if sibling is best_elem:
                append = True
            sibling_key = sibling  # HashableElement(sibling)
            if (
                sibling_key in candidates
                and candidates[sibling_key]["content_score"] >= sibling_score_threshold
            ):
                append = True

            if sibling.tag == "p":
                link_density = self.get_link_density(sibling)
                node_content = sibling.text or ""
                node_length = len(node_content)

                if node_length > 80 and link_density < 0.25:
                    append = True
                elif (
                    node_length <= 80
                    and link_density == 0
                    and re.search(r"\.( |$)", node_content)
                ):
                    append = True

            if append:
                # We don't want to append directly to output, but the div
                # in html->body->div
                if html_partial:
                    output.append(sibling)
                else:
                    output.getchildren()[0].getchildren()[0].append(sibling)
        # if output is not None:
        #    output.append(best_elem)
        return output

    def select_best_candidate(self, candidates):
        if not candidates:
            return None

        sorted_candidates = sorted(
            candidates.values(), key=lambda x: x["content_score"], reverse=True
        )
        for candidate in sorted_candidates[:5]:
            elem = candidate["elem"]
            log.debug("Top 5 : %6.3f %s" % (candidate["content_score"], describe(elem)))

        best_candidate = sorted_candidates[0]
        return best_candidate

    def get_link_density(self, elem):
        link_length = 0
        for i in elem.findall(".//a"):
            link_length += text_length(i)
        # if len(elem.findall(".//div") or elem.findall(".//p")):
        #    link_length = link_length
        total_length = text_length(elem)
        return float(link_length) / max(total_length, 1)

    def score_paragraphs(self):
        MIN_LEN = self.min_text_length
        candidates = {}
        ordered = []
        for elem in self.tags(self._html(), "p", "pre", "td"):
            parent_node = elem.getparent()
            if parent_node is None:
                continue
            grand_parent_node = parent_node.getparent()

            inner_text = clean(elem.text_content() or "")
            inner_text_len = len(inner_text)

            # If this paragraph is less than 25 characters
            # don't even count it.
            if inner_text_len < MIN_LEN:
                continue

            if parent_node not in candidates:
                candidates[parent_node] = self.score_node(parent_node)
                ordered.append(parent_node)

            if grand_parent_node is not None and grand_parent_node not in candidates:
                candidates[grand_parent_node] = self.score_node(grand_parent_node)
                ordered.append(grand_parent_node)

            content_score = 1
            content_score += len(inner_text.split(","))
            content_score += min((inner_text_len / 100), 3)
            # if elem not in candidates:
            #    candidates[elem] = self.score_node(elem)

            # WTF? candidates[elem]['content_score'] += content_score
            candidates[parent_node]["content_score"] += content_score
            if grand_parent_node is not None:
                candidates[grand_parent_node]["content_score"] += content_score / 2.0

        # Scale the final candidates score based on link density. Good content
        # should have a relatively small link density (5% or less) and be
        # mostly unaffected by this operation.
        for elem in ordered:
            candidate = candidates[elem]
            ld = self.get_link_density(elem)
            score = candidate["content_score"]
            log.debug(
                "Branch %6.3f %s link density %.3f -> %6.3f"
                % (score, describe(elem), ld, score * (1 - ld))
            )
            candidate["content_score"] *= 1 - ld

        return candidates

    def class_weight(self, e):
        weight = 0
        for feature in [e.get("class", None), e.get("id", None)]:
            if feature:
                if REGEXES["negativeRe"].search(feature):
                    weight -= 25

                if REGEXES["positiveRe"].search(feature):
                    weight += 25

                if self.positive_keywords and self.positive_keywords.search(feature):
                    weight += 25

                if self.negative_keywords and self.negative_keywords.search(feature):
                    weight -= 25

        if self.positive_keywords and self.positive_keywords.match("tag-" + e.tag):
            weight += 25

        if self.negative_keywords and self.negative_keywords.match("tag-" + e.tag):
            weight -= 25

        return weight

    def score_node(self, elem):
        content_score = self.class_weight(elem)
        name = elem.tag.lower()
        if name in ["div", "article"]:
            content_score += 5
        elif name in ["pre", "td", "blockquote"]:
            content_score += 3
        elif name in ["address", "ol", "ul", "dl", "dd", "dt", "li", "form", "aside"]:
            content_score -= 3
        elif name in [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "th",
            "header",
            "footer",
            "nav",
        ]:
            content_score -= 5
        return {"content_score": content_score, "elem": elem}

    def remove_unlikely_candidates(self):
        for elem in self.html.findall(".//*"):
            s = "%s %s" % (elem.get("class", ""), elem.get("id", ""))
            if len(s) < 2:
                continue
            if (
                REGEXES["unlikelyCandidatesRe"].search(s)
                and (not REGEXES["okMaybeItsACandidateRe"].search(s))
                and elem.tag not in ["html", "body"]
            ):
                log.debug("Removing unlikely candidate - %s" % describe(elem))
                elem.drop_tree()

    def transform_misused_divs_into_paragraphs(self):
        for elem in self.tags(self.html, "div"):
            # transform <div>s that do not contain other block elements into
            # <p>s
            # FIXME: The current implementation ignores all descendants that
            # are not direct children of elem
            # This results in incorrect results in case there is an <img>
            # buried within an <a> for example
            if not REGEXES["divToPElementsRe"].search(
                str_(b"".join(map(tostring_, list(elem))))
            ):
                # log.debug("Altering %s to p" % (describe(elem)))
                elem.tag = "p"
                # print "Fixed element "+describe(elem)

        for elem in self.tags(self.html, "div"):
            if elem.text and elem.text.strip():
                p = fragment_fromstring("<p/>")
                p.text = elem.text
                elem.text = None
                elem.insert(0, p)
                # print "Appended "+tounicode(p)+" to "+describe(elem)

            for pos, child in reversed(list(enumerate(elem))):
                if child.tail and child.tail.strip():
                    p = fragment_fromstring("<p/>")
                    p.text = child.tail
                    child.tail = None
                    elem.insert(pos + 1, p)
                    # print "Inserted "+tounicode(p)+" to "+describe(elem)
                if child.tag == "br":
                    # print 'Dropped <br> at '+describe(elem)
                    child.drop_tree()

    def tags(self, node, *tag_names):
        for tag_name in tag_names:
            for e in node.findall(".//%s" % tag_name):
                yield e

    def reverse_tags(self, node, *tag_names):
        for tag_name in tag_names:
            for e in reversed(node.findall(".//%s" % tag_name)):
                yield e

    def sanitize(self, node, candidates):
        MIN_LEN = self.min_text_length
        for header in self.tags(node, "h1", "h2", "h3", "h4", "h5", "h6"):
            if self.class_weight(header) < 0 or self.get_link_density(header) > 0.33:
                header.drop_tree()

        for elem in self.tags(node, "form", "textarea"):
            elem.drop_tree()

        for elem in self.tags(node, "iframe"):
            if "src" in elem.attrib and REGEXES["videoRe"].search(elem.attrib["src"]):
                elem.text = "VIDEO"  # ADD content to iframe text node to force <iframe></iframe> proper output
            else:
                elem.drop_tree()

        allowed = {}
        # Conditionally clean <table>s, <ul>s, and <div>s
        for el in self.reverse_tags(
            node, "table", "ul", "div", "aside", "header", "footer", "section"
        ):
            if el in allowed:
                continue
            weight = self.class_weight(el)
            if el in candidates:
                content_score = candidates[el]["content_score"]
                # print '!',el, '-> %6.3f' % content_score
            else:
                content_score = 0
            tag = el.tag

            if weight + content_score < 0:
                log.debug(
                    "Removed %s with score %6.3f and weight %-3s"
                    % (describe(el), content_score, weight,)
                )
                el.drop_tree()
            elif el.text_content().count(",") < 10:
                counts = {}
                for kind in ["p", "img", "li", "a", "embed", "input"]:
                    counts[kind] = len(el.findall(".//%s" % kind))
                counts["li"] -= 100
                counts["input"] -= len(el.findall('.//input[@type="hidden"]'))

                # Count the text length excluding any surrounding whitespace
                content_length = text_length(el)
                link_density = self.get_link_density(el)
                parent_node = el.getparent()
                if parent_node is not None:
                    if parent_node in candidates:
                        content_score = candidates[parent_node]["content_score"]
                    else:
                        content_score = 0
                # if parent_node is not None:
                # pweight = self.class_weight(parent_node) + content_score
                # pname = describe(parent_node)
                # else:
                # pweight = 0
                # pname = "no parent"
                to_remove = False
                reason = ""

                # if el.tag == 'div' and counts["img"] >= 1:
                #    continue
                if counts["p"] and counts["img"] > 1 + counts["p"] * 1.3:
                    reason = "too many images (%s)" % counts["img"]
                    to_remove = True
                elif counts["li"] > counts["p"] and tag not in ("ol", "ul"):
                    reason = "more <li>s than <p>s"
                    to_remove = True
                elif counts["input"] > (counts["p"] / 3):
                    reason = "less than 3x <p>s than <input>s"
                    to_remove = True
                elif content_length < MIN_LEN and counts["img"] == 0:
                    reason = (
                        "too short content length %s without a single image"
                        % content_length
                    )
                    to_remove = True
                elif content_length < MIN_LEN and counts["img"] > 2:
                    reason = (
                        "too short content length %s and too many images"
                        % content_length
                    )
                    to_remove = True
                elif weight < 25 and link_density > 0.2:
                    reason = "too many links %.3f for its weight %s" % (
                        link_density,
                        weight,
                    )
                    to_remove = True
                elif weight >= 25 and link_density > 0.5:
                    reason = "too many links %.3f for its weight %s" % (
                        link_density,
                        weight,
                    )
                    to_remove = True
                elif (counts["embed"] == 1 and content_length < 75) or counts[
                    "embed"
                ] > 1:
                    reason = (
                        "<embed>s with too short content length, or too many <embed>s"
                    )
                    to_remove = True
                elif not content_length:
                    reason = "no content"
                    to_remove = True
                    #                if el.tag == 'div' and counts['img'] >= 1 and to_remove:
                    #                    imgs = el.findall('.//img')
                    #                    valid_img = False
                    #                    log.debug(tounicode(el))
                    #                    for img in imgs:
                    #
                    #                        height = img.get('height')
                    #                        text_length = img.get('text_length')
                    #                        log.debug ("height %s text_length %s" %(repr(height), repr(text_length)))
                    #                        if to_int(height) >= 100 or to_int(text_length) >= 100:
                    #                            valid_img = True
                    #                            log.debug("valid image" + tounicode(img))
                    #                            break
                    #                    if valid_img:
                    #                        to_remove = False
                    #                        log.debug("Allowing %s" %el.text_content())
                    #                        for desnode in self.tags(el, "table", "ul", "div"):
                    #                            allowed[desnode] = True

                    # find x non empty preceding and succeeding siblings
                    i, j = 0, 0
                    x = 1
                    siblings = []
                    for sib in el.itersiblings():
                        # log.debug(sib.text_content())
                        sib_content_length = text_length(sib)
                        if sib_content_length:
                            i = +1
                            siblings.append(sib_content_length)
                            if i == x:
                                break
                    for sib in el.itersiblings(preceding=True):
                        # log.debug(sib.text_content())
                        sib_content_length = text_length(sib)
                        if sib_content_length:
                            j = +1
                            siblings.append(sib_content_length)
                            if j == x:
                                break
                    # log.debug(str_(siblings))
                    if siblings and sum(siblings) > 1000:
                        to_remove = False
                        log.debug("Allowing %s" % describe(el))
                        for desnode in self.tags(el, "table", "ul", "div", "section"):
                            allowed[desnode] = True

                if to_remove:
                    log.debug(
                        "Removed %6.3f %s with weight %s cause it has %s."
                        % (content_score, describe(el), weight, reason)
                    )
                    # print tounicode(el)
                    # log.debug("pname %s pweight %.3f" %(pname, pweight))
                    el.drop_tree()
                else:
                    log.debug(
                        "Not removing %s of length %s: %s"
                        % (describe(el), content_length, text_content(el))
                    )

        self.html = node
        return self.get_clean_html()


def main():
    VERBOSITY = {1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}

    from optparse import OptionParser

    parser = OptionParser(usage="%prog: [options] [file]")
    parser.add_option("-v", "--verbose", action="count", default=0)
    parser.add_option(
        "-b", "--browser", default=None, action="store_true", help="open in browser"
    )
    parser.add_option(
        "-l", "--log", default=None, help="save logs into file (appended)"
    )
    parser.add_option(
        "-u", "--url", default=None, help="use URL instead of a local file"
    )
    parser.add_option("-x", "--xpath", default=None, help="add original xpath")
    parser.add_option(
        "-p",
        "--positive-keywords",
        default=None,
        help="positive keywords (comma-separated)",
        action="store",
    )
    parser.add_option(
        "-n",
        "--negative-keywords",
        default=None,
        help="negative keywords (comma-separated)",
        action="store",
    )
    (options, args) = parser.parse_args()

    if options.verbose:
        logging.basicConfig(
            level=VERBOSITY[options.verbose],
            filename=options.log,
            format="%(asctime)s: %(levelname)s: %(message)s (at %(filename)s: %(lineno)d)",
        )

    if not (len(args) == 1 or options.url):
        parser.print_help()
        sys.exit(1)

    file = None
    if options.url:
        headers = {"User-Agent": "Mozilla/5.0"}
        if sys.version_info[0] == 3:
            import urllib.request, urllib.parse, urllib.error

            request = urllib.request.Request(options.url, None, headers)
            file = urllib.request.urlopen(request)
        else:
            import urllib2

            request = urllib2.Request(options.url, None, headers)
            file = urllib2.urlopen(request)
    else:
        file = open(args[0], "rt")
    try:
        doc = Document(
            file.read(),
            url=options.url,
            positive_keywords=options.positive_keywords,
            negative_keywords=options.negative_keywords,
        )
        if options.browser:
            from .browser import open_in_browser

            result = "<h2>" + doc.short_title() + "</h2><br/>" + doc.summary()
            open_in_browser(result)
        else:
            enc = (
                sys.__stdout__.encoding or "utf-8"
            )  # XXX: this hack could not always work, better to set PYTHONIOENCODING
            result = "Title:" + doc.short_title() + "\n" + doc.summary()
            if sys.version_info[0] == 3:
                print(result)
            else:
                print(result.encode(enc, "replace"))
    finally:
        file.close()


if __name__ == "__main__":
    main()
