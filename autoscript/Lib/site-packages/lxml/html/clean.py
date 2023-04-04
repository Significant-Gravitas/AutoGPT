# cython: language_level=3str

"""A cleanup tool for HTML.

Removes unwanted tags and content.  See the `Cleaner` class for
details.
"""

from __future__ import absolute_import

import copy
import re
import sys
try:
    from urlparse import urlsplit
    from urllib import unquote_plus
except ImportError:
    # Python 3
    from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result

try:
    unichr
except NameError:
    # Python 3
    unichr = chr
try:
    unicode
except NameError:
    # Python 3
    unicode = str
try:
    basestring
except NameError:
    basestring = (str, bytes)


__all__ = ['clean_html', 'clean', 'Cleaner', 'autolink', 'autolink_html',
           'word_break', 'word_break_html']

# Look at http://code.sixapart.com/trac/livejournal/browser/trunk/cgi-bin/cleanhtml.pl
#   Particularly the CSS cleaning; most of the tag cleaning is integrated now
# I have multiple kinds of schemes searched; but should schemes be
#   whitelisted instead?
# max height?
# remove images?  Also in CSS?  background attribute?
# Some way to whitelist object, iframe, etc (e.g., if you want to
#   allow *just* embedded YouTube movies)
# Log what was deleted and why?
# style="behavior: ..." might be bad in IE?
# Should we have something for just <meta http-equiv>?  That's the worst of the
#   metas.
# UTF-7 detections?  Example:
#     <HEAD><META HTTP-EQUIV="CONTENT-TYPE" CONTENT="text/html; charset=UTF-7"> </HEAD>+ADw-SCRIPT+AD4-alert('XSS');+ADw-/SCRIPT+AD4-
#   you don't always have to have the charset set, if the page has no charset
#   and there's UTF7-like code in it.
# Look at these tests: http://htmlpurifier.org/live/smoketests/xssAttacks.php


# This is an IE-specific construct you can have in a stylesheet to
# run some Javascript:
_replace_css_javascript = re.compile(
    r'expression\s*\(.*?\)', re.S|re.I).sub

# Do I have to worry about @\nimport?
_replace_css_import = re.compile(
    r'@\s*import', re.I).sub

_looks_like_tag_content = re.compile(
    r'</?[a-zA-Z]+|\son[a-zA-Z]+\s*=',
    *((re.ASCII,) if sys.version_info[0] >= 3 else ())).search

# All kinds of schemes besides just javascript: that can cause
# execution:
_find_image_dataurls = re.compile(
    r'data:image/(.+);base64,', re.I).findall
_possibly_malicious_schemes = re.compile(
    r'(javascript|jscript|livescript|vbscript|data|about|mocha):',
    re.I).findall
# SVG images can contain script content
_is_unsafe_image_type = re.compile(r"(xml|svg)", re.I).search

def _has_javascript_scheme(s):
    safe_image_urls = 0
    for image_type in _find_image_dataurls(s):
        if _is_unsafe_image_type(image_type):
            return True
        safe_image_urls += 1
    return len(_possibly_malicious_schemes(s)) > safe_image_urls

_substitute_whitespace = re.compile(r'[\s\x00-\x08\x0B\x0C\x0E-\x19]+').sub

# FIXME: check against: http://msdn2.microsoft.com/en-us/library/ms537512.aspx
_conditional_comment_re = re.compile(
    r'\[if[\s\n\r]+.*?][\s\n\r]*>', re.I|re.S)

_find_styled_elements = etree.XPath(
    "descendant-or-self::*[@style]")

_find_external_links = etree.XPath(
    ("descendant-or-self::a  [normalize-space(@href) and substring(normalize-space(@href),1,1) != '#'] |"
     "descendant-or-self::x:a[normalize-space(@href) and substring(normalize-space(@href),1,1) != '#']"),
    namespaces={'x':XHTML_NAMESPACE})


class Cleaner(object):
    """
    Instances cleans the document of each of the possible offending
    elements.  The cleaning is controlled by attributes; you can
    override attributes in a subclass, or set them in the constructor.

    ``scripts``:
        Removes any ``<script>`` tags.

    ``javascript``:
        Removes any Javascript, like an ``onclick`` attribute. Also removes stylesheets
        as they could contain Javascript.

    ``comments``:
        Removes any comments.

    ``style``:
        Removes any style tags.

    ``inline_style``
        Removes any style attributes.  Defaults to the value of the ``style`` option.

    ``links``:
        Removes any ``<link>`` tags

    ``meta``:
        Removes any ``<meta>`` tags

    ``page_structure``:
        Structural parts of a page: ``<head>``, ``<html>``, ``<title>``.

    ``processing_instructions``:
        Removes any processing instructions.

    ``embedded``:
        Removes any embedded objects (flash, iframes)

    ``frames``:
        Removes any frame-related tags

    ``forms``:
        Removes any form tags

    ``annoying_tags``:
        Tags that aren't *wrong*, but are annoying.  ``<blink>`` and ``<marquee>``

    ``remove_tags``:
        A list of tags to remove.  Only the tags will be removed,
        their content will get pulled up into the parent tag.

    ``kill_tags``:
        A list of tags to kill.  Killing also removes the tag's content,
        i.e. the whole subtree, not just the tag itself.

    ``allow_tags``:
        A list of tags to include (default include all).

    ``remove_unknown_tags``:
        Remove any tags that aren't standard parts of HTML.

    ``safe_attrs_only``:
        If true, only include 'safe' attributes (specifically the list
        from the feedparser HTML sanitisation web site).

    ``safe_attrs``:
        A set of attribute names to override the default list of attributes
        considered 'safe' (when safe_attrs_only=True).

    ``add_nofollow``:
        If true, then any <a> tags will have ``rel="nofollow"`` added to them.

    ``host_whitelist``:
        A list or set of hosts that you can use for embedded content
        (for content like ``<object>``, ``<link rel="stylesheet">``, etc).
        You can also implement/override the method
        ``allow_embedded_url(el, url)`` or ``allow_element(el)`` to
        implement more complex rules for what can be embedded.
        Anything that passes this test will be shown, regardless of
        the value of (for instance) ``embedded``.

        Note that this parameter might not work as intended if you do not
        make the links absolute before doing the cleaning.

        Note that you may also need to set ``whitelist_tags``.

    ``whitelist_tags``:
        A set of tags that can be included with ``host_whitelist``.
        The default is ``iframe`` and ``embed``; you may wish to
        include other tags like ``script``, or you may want to
        implement ``allow_embedded_url`` for more control.  Set to None to
        include all tags.

    This modifies the document *in place*.
    """

    scripts = True
    javascript = True
    comments = True
    style = False
    inline_style = None
    links = True
    meta = True
    page_structure = True
    processing_instructions = True
    embedded = True
    frames = True
    forms = True
    annoying_tags = True
    remove_tags = None
    allow_tags = None
    kill_tags = None
    remove_unknown_tags = True
    safe_attrs_only = True
    safe_attrs = defs.safe_attrs
    add_nofollow = False
    host_whitelist = ()
    whitelist_tags = {'iframe', 'embed'}

    def __init__(self, **kw):
        not_an_attribute = object()
        for name, value in kw.items():
            default = getattr(self, name, not_an_attribute)
            if (default is not None and default is not True and default is not False
                    and not isinstance(default, (frozenset, set, tuple, list))):
                raise TypeError(
                    "Unknown parameter: %s=%r" % (name, value))
            setattr(self, name, value)
        if self.inline_style is None and 'inline_style' not in kw:
            self.inline_style = self.style

        if kw.get("allow_tags"):
            if kw.get("remove_unknown_tags"):
                raise ValueError("It does not make sense to pass in both "
                                 "allow_tags and remove_unknown_tags")
            self.remove_unknown_tags = False

    # Used to lookup the primary URL for a given tag that is up for
    # removal:
    _tag_link_attrs = dict(
        script='src',
        link='href',
        # From: http://java.sun.com/j2se/1.4.2/docs/guide/misc/applet.html
        # From what I can tell, both attributes can contain a link:
        applet=['code', 'object'],
        iframe='src',
        embed='src',
        layer='src',
        # FIXME: there doesn't really seem like a general way to figure out what
        # links an <object> tag uses; links often go in <param> tags with values
        # that we don't really know.  You'd have to have knowledge about specific
        # kinds of plugins (probably keyed off classid), and match against those.
        ##object=?,
        # FIXME: not looking at the action currently, because it is more complex
        # than than -- if you keep the form, you should keep the form controls.
        ##form='action',
        a='href',
        )

    def __call__(self, doc):
        """
        Cleans the document.
        """
        try:
            getroot = doc.getroot
        except AttributeError:
            pass  # Element instance
        else:
            doc = getroot()  # ElementTree instance, instead of an element
        # convert XHTML to HTML
        xhtml_to_html(doc)
        # Normalize a case that IE treats <image> like <img>, and that
        # can confuse either this step or later steps.
        for el in doc.iter('image'):
            el.tag = 'img'
        if not self.comments:
            # Of course, if we were going to kill comments anyway, we don't
            # need to worry about this
            self.kill_conditional_comments(doc)

        kill_tags = set(self.kill_tags or ())
        remove_tags = set(self.remove_tags or ())
        allow_tags = set(self.allow_tags or ())

        if self.scripts:
            kill_tags.add('script')
        if self.safe_attrs_only:
            safe_attrs = set(self.safe_attrs)
            for el in doc.iter(etree.Element):
                attrib = el.attrib
                for aname in attrib.keys():
                    if aname not in safe_attrs:
                        del attrib[aname]
        if self.javascript:
            if not (self.safe_attrs_only and
                    self.safe_attrs == defs.safe_attrs):
                # safe_attrs handles events attributes itself
                for el in doc.iter(etree.Element):
                    attrib = el.attrib
                    for aname in attrib.keys():
                        if aname.startswith('on'):
                            del attrib[aname]
            doc.rewrite_links(self._remove_javascript_link,
                              resolve_base_href=False)
            # If we're deleting style then we don't have to remove JS links
            # from styles, otherwise...
            if not self.inline_style:
                for el in _find_styled_elements(doc):
                    old = el.get('style')
                    new = _replace_css_javascript('', old)
                    new = _replace_css_import('', new)
                    if self._has_sneaky_javascript(new):
                        # Something tricky is going on...
                        del el.attrib['style']
                    elif new != old:
                        el.set('style', new)
            if not self.style:
                for el in list(doc.iter('style')):
                    if el.get('type', '').lower().strip() == 'text/javascript':
                        el.drop_tree()
                        continue
                    old = el.text or ''
                    new = _replace_css_javascript('', old)
                    # The imported CSS can do anything; we just can't allow:
                    new = _replace_css_import('', new)
                    if self._has_sneaky_javascript(new):
                        # Something tricky is going on...
                        el.text = '/* deleted */'
                    elif new != old:
                        el.text = new
        if self.comments:
            kill_tags.add(etree.Comment)
        if self.processing_instructions:
            kill_tags.add(etree.ProcessingInstruction)
        if self.style:
            kill_tags.add('style')
        if self.inline_style:
            etree.strip_attributes(doc, 'style')
        if self.links:
            kill_tags.add('link')
        elif self.style or self.javascript:
            # We must get rid of included stylesheets if Javascript is not
            # allowed, as you can put Javascript in them
            for el in list(doc.iter('link')):
                if 'stylesheet' in el.get('rel', '').lower():
                    # Note this kills alternate stylesheets as well
                    if not self.allow_element(el):
                        el.drop_tree()
        if self.meta:
            kill_tags.add('meta')
        if self.page_structure:
            remove_tags.update(('head', 'html', 'title'))
        if self.embedded:
            # FIXME: is <layer> really embedded?
            # We should get rid of any <param> tags not inside <applet>;
            # These are not really valid anyway.
            for el in list(doc.iter('param')):
                parent = el.getparent()
                while parent is not None and parent.tag not in ('applet', 'object'):
                    parent = parent.getparent()
                if parent is None:
                    el.drop_tree()
            kill_tags.update(('applet',))
            # The alternate contents that are in an iframe are a good fallback:
            remove_tags.update(('iframe', 'embed', 'layer', 'object', 'param'))
        if self.frames:
            # FIXME: ideally we should look at the frame links, but
            # generally frames don't mix properly with an HTML
            # fragment anyway.
            kill_tags.update(defs.frame_tags)
        if self.forms:
            remove_tags.add('form')
            kill_tags.update(('button', 'input', 'select', 'textarea'))
        if self.annoying_tags:
            remove_tags.update(('blink', 'marquee'))

        _remove = []
        _kill = []
        for el in doc.iter():
            if el.tag in kill_tags:
                if self.allow_element(el):
                    continue
                _kill.append(el)
            elif el.tag in remove_tags:
                if self.allow_element(el):
                    continue
                _remove.append(el)

        if _remove and _remove[0] == doc:
            # We have to drop the parent-most tag, which we can't
            # do.  Instead we'll rewrite it:
            el = _remove.pop(0)
            el.tag = 'div'
            el.attrib.clear()
        elif _kill and _kill[0] == doc:
            # We have to drop the parent-most element, which we can't
            # do.  Instead we'll clear it:
            el = _kill.pop(0)
            if el.tag != 'html':
                el.tag = 'div'
            el.clear()

        _kill.reverse() # start with innermost tags
        for el in _kill:
            el.drop_tree()
        for el in _remove:
            el.drop_tag()

        if self.remove_unknown_tags:
            if allow_tags:
                raise ValueError(
                    "It does not make sense to pass in both allow_tags and remove_unknown_tags")
            allow_tags = set(defs.tags)
        if allow_tags:
            # make sure we do not remove comments/PIs if users want them (which is rare enough)
            if not self.comments:
                allow_tags.add(etree.Comment)
            if not self.processing_instructions:
                allow_tags.add(etree.ProcessingInstruction)

            bad = []
            for el in doc.iter():
                if el.tag not in allow_tags:
                    bad.append(el)
            if bad:
                if bad[0] is doc:
                    el = bad.pop(0)
                    el.tag = 'div'
                    el.attrib.clear()
                for el in bad:
                    el.drop_tag()
        if self.add_nofollow:
            for el in _find_external_links(doc):
                if not self.allow_follow(el):
                    rel = el.get('rel')
                    if rel:
                        if ('nofollow' in rel
                                and ' nofollow ' in (' %s ' % rel)):
                            continue
                        rel = '%s nofollow' % rel
                    else:
                        rel = 'nofollow'
                    el.set('rel', rel)

    def allow_follow(self, anchor):
        """
        Override to suppress rel="nofollow" on some anchors.
        """
        return False

    def allow_element(self, el):
        """
        Decide whether an element is configured to be accepted or rejected.

        :param el: an element.
        :return: true to accept the element or false to reject/discard it.
        """
        if el.tag not in self._tag_link_attrs:
            return False
        attr = self._tag_link_attrs[el.tag]
        if isinstance(attr, (list, tuple)):
            for one_attr in attr:
                url = el.get(one_attr)
                if not url:
                    return False
                if not self.allow_embedded_url(el, url):
                    return False
            return True
        else:
            url = el.get(attr)
            if not url:
                return False
            return self.allow_embedded_url(el, url)

    def allow_embedded_url(self, el, url):
        """
        Decide whether a URL that was found in an element's attributes or text
        if configured to be accepted or rejected.

        :param el: an element.
        :param url: a URL found on the element.
        :return: true to accept the URL and false to reject it.
        """
        if self.whitelist_tags is not None and el.tag not in self.whitelist_tags:
            return False
        scheme, netloc, path, query, fragment = urlsplit(url)
        netloc = netloc.lower().split(':', 1)[0]
        if scheme not in ('http', 'https'):
            return False
        if netloc in self.host_whitelist:
            return True
        return False

    def kill_conditional_comments(self, doc):
        """
        IE conditional comments basically embed HTML that the parser
        doesn't normally see.  We can't allow anything like that, so
        we'll kill any comments that could be conditional.
        """
        has_conditional_comment = _conditional_comment_re.search
        self._kill_elements(
            doc, lambda el: has_conditional_comment(el.text),
            etree.Comment)                

    def _kill_elements(self, doc, condition, iterate=None):
        bad = []
        for el in doc.iter(iterate):
            if condition(el):
                bad.append(el)
        for el in bad:
            el.drop_tree()

    def _remove_javascript_link(self, link):
        # links like "j a v a s c r i p t:" might be interpreted in IE
        new = _substitute_whitespace('', unquote_plus(link))
        if _has_javascript_scheme(new):
            # FIXME: should this be None to delete?
            return ''
        return link

    _substitute_comments = re.compile(r'/\*.*?\*/', re.S).sub

    def _has_sneaky_javascript(self, style):
        """
        Depending on the browser, stuff like ``e x p r e s s i o n(...)``
        can get interpreted, or ``expre/* stuff */ssion(...)``.  This
        checks for attempt to do stuff like this.

        Typically the response will be to kill the entire style; if you
        have just a bit of Javascript in the style another rule will catch
        that and remove only the Javascript from the style; this catches
        more sneaky attempts.
        """
        style = self._substitute_comments('', style)
        style = style.replace('\\', '')
        style = _substitute_whitespace('', style)
        style = style.lower()
        if _has_javascript_scheme(style):
            return True
        if 'expression(' in style:
            return True
        if '@import' in style:
            return True
        if '</noscript' in style:
            # e.g. '<noscript><style><a title="</noscript><img src=x onerror=alert(1)>">'
            return True
        if _looks_like_tag_content(style):
            # e.g. '<math><style><img src=x onerror=alert(1)></style></math>'
            return True
        return False

    def clean_html(self, html):
        result_type = type(html)
        if isinstance(html, basestring):
            doc = fromstring(html)
        else:
            doc = copy.deepcopy(html)
        self(doc)
        return _transform_result(result_type, doc)

clean = Cleaner()
clean_html = clean.clean_html

############################################################
## Autolinking
############################################################

_link_regexes = [
    re.compile(r'(?P<body>https?://(?P<host>[a-z0-9._-]+)(?:/[/\-_.,a-z0-9%&?;=~]*)?(?:\([/\-_.,a-z0-9%&?;=~]*\))?)', re.I),
    # This is conservative, but autolinking can be a bit conservative:
    re.compile(r'mailto:(?P<body>[a-z0-9._-]+@(?P<host>[a-z0-9_.-]+[a-z]))', re.I),
    ]

_avoid_elements = ['textarea', 'pre', 'code', 'head', 'select', 'a']

_avoid_hosts = [
    re.compile(r'^localhost', re.I),
    re.compile(r'\bexample\.(?:com|org|net)$', re.I),
    re.compile(r'^127\.0\.0\.1$'),
    ]

_avoid_classes = ['nolink']

def autolink(el, link_regexes=_link_regexes,
             avoid_elements=_avoid_elements,
             avoid_hosts=_avoid_hosts,
             avoid_classes=_avoid_classes):
    """
    Turn any URLs into links.

    It will search for links identified by the given regular
    expressions (by default mailto and http(s) links).

    It won't link text in an element in avoid_elements, or an element
    with a class in avoid_classes.  It won't link to anything with a
    host that matches one of the regular expressions in avoid_hosts
    (default localhost and 127.0.0.1).

    If you pass in an element, the element's tail will not be
    substituted, only the contents of the element.
    """
    if el.tag in avoid_elements:
        return
    class_name = el.get('class')
    if class_name:
        class_name = class_name.split()
        for match_class in avoid_classes:
            if match_class in class_name:
                return
    for child in list(el):
        autolink(child, link_regexes=link_regexes,
                 avoid_elements=avoid_elements,
                 avoid_hosts=avoid_hosts,
                 avoid_classes=avoid_classes)
        if child.tail:
            text, tail_children = _link_text(
                child.tail, link_regexes, avoid_hosts, factory=el.makeelement)
            if tail_children:
                child.tail = text
                index = el.index(child)
                el[index+1:index+1] = tail_children
    if el.text:
        text, pre_children = _link_text(
            el.text, link_regexes, avoid_hosts, factory=el.makeelement)
        if pre_children:
            el.text = text
            el[:0] = pre_children

def _link_text(text, link_regexes, avoid_hosts, factory):
    leading_text = ''
    links = []
    last_pos = 0
    while 1:
        best_match, best_pos = None, None
        for regex in link_regexes:
            regex_pos = last_pos
            while 1:
                match = regex.search(text, pos=regex_pos)
                if match is None:
                    break
                host = match.group('host')
                for host_regex in avoid_hosts:
                    if host_regex.search(host):
                        regex_pos = match.end()
                        break
                else:
                    break
            if match is None:
                continue
            if best_pos is None or match.start() < best_pos:
                best_match = match
                best_pos = match.start()
        if best_match is None:
            # No more matches
            if links:
                assert not links[-1].tail
                links[-1].tail = text
            else:
                assert not leading_text
                leading_text = text
            break
        link = best_match.group(0)
        end = best_match.end()
        if link.endswith('.') or link.endswith(','):
            # These punctuation marks shouldn't end a link
            end -= 1
            link = link[:-1]
        prev_text = text[:best_match.start()]
        if links:
            assert not links[-1].tail
            links[-1].tail = prev_text
        else:
            assert not leading_text
            leading_text = prev_text
        anchor = factory('a')
        anchor.set('href', link)
        body = best_match.group('body')
        if not body:
            body = link
        if body.endswith('.') or body.endswith(','):
            body = body[:-1]
        anchor.text = body
        links.append(anchor)
        text = text[end:]
    return leading_text, links
                
def autolink_html(html, *args, **kw):
    result_type = type(html)
    if isinstance(html, basestring):
        doc = fromstring(html)
    else:
        doc = copy.deepcopy(html)
    autolink(doc, *args, **kw)
    return _transform_result(result_type, doc)

autolink_html.__doc__ = autolink.__doc__

############################################################
## Word wrapping
############################################################

_avoid_word_break_elements = ['pre', 'textarea', 'code']
_avoid_word_break_classes = ['nobreak']

def word_break(el, max_width=40,
               avoid_elements=_avoid_word_break_elements,
               avoid_classes=_avoid_word_break_classes,
               break_character=unichr(0x200b)):
    """
    Breaks any long words found in the body of the text (not attributes).

    Doesn't effect any of the tags in avoid_elements, by default
    ``<textarea>`` and ``<pre>``

    Breaks words by inserting &#8203;, which is a unicode character
    for Zero Width Space character.  This generally takes up no space
    in rendering, but does copy as a space, and in monospace contexts
    usually takes up space.

    See http://www.cs.tut.fi/~jkorpela/html/nobr.html for a discussion
    """
    # Character suggestion of &#8203 comes from:
    #   http://www.cs.tut.fi/~jkorpela/html/nobr.html
    if el.tag in _avoid_word_break_elements:
        return
    class_name = el.get('class')
    if class_name:
        dont_break = False
        class_name = class_name.split()
        for avoid in avoid_classes:
            if avoid in class_name:
                dont_break = True
                break
        if dont_break:
            return
    if el.text:
        el.text = _break_text(el.text, max_width, break_character)
    for child in el:
        word_break(child, max_width=max_width,
                   avoid_elements=avoid_elements,
                   avoid_classes=avoid_classes,
                   break_character=break_character)
        if child.tail:
            child.tail = _break_text(child.tail, max_width, break_character)

def word_break_html(html, *args, **kw):
    result_type = type(html)
    doc = fromstring(html)
    word_break(doc, *args, **kw)
    return _transform_result(result_type, doc)

def _break_text(text, max_width, break_character):
    words = text.split()
    for word in words:
        if len(word) > max_width:
            replacement = _insert_break(word, max_width, break_character)
            text = text.replace(word, replacement)
    return text

_break_prefer_re = re.compile(r'[^a-z]', re.I)

def _insert_break(word, width, break_character):
    orig_word = word
    result = ''
    while len(word) > width:
        start = word[:width]
        breaks = list(_break_prefer_re.finditer(start))
        if breaks:
            last_break = breaks[-1]
            # Only walk back up to 10 characters to find a nice break:
            if last_break.end() > width-10:
                # FIXME: should the break character be at the end of the
                # chunk, or the beginning of the next chunk?
                start = word[:last_break.end()]
        result += start + break_character
        word = word[len(start):]
    result += word
    return result
    
