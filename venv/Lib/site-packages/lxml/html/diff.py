# cython: language_level=3

from __future__ import absolute_import

import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re

__all__ = ['html_annotate', 'htmldiff']

try:
    from html import escape as html_escape
except ImportError:
    from cgi import escape as html_escape
try:
    _unicode = unicode
except NameError:
    # Python 3
    _unicode = str
try:
    basestring
except NameError:
    # Python 3
    basestring = str

############################################################
## Annotation
############################################################

def default_markup(text, version):
    return '<span title="%s">%s</span>' % (
        html_escape(_unicode(version), 1), text)

def html_annotate(doclist, markup=default_markup):
    """
    doclist should be ordered from oldest to newest, like::

        >>> version1 = 'Hello World'
        >>> version2 = 'Goodbye World'
        >>> print(html_annotate([(version1, 'version 1'),
        ...                      (version2, 'version 2')]))
        <span title="version 2">Goodbye</span> <span title="version 1">World</span>

    The documents must be *fragments* (str/UTF8 or unicode), not
    complete documents

    The markup argument is a function to markup the spans of words.
    This function is called like markup('Hello', 'version 2'), and
    returns HTML.  The first argument is text and never includes any
    markup.  The default uses a span with a title:

        >>> print(default_markup('Some Text', 'by Joe'))
        <span title="by Joe">Some Text</span>
    """
    # The basic strategy we have is to split the documents up into
    # logical tokens (which are words with attached markup).  We then
    # do diffs of each of the versions to track when a token first
    # appeared in the document; the annotation attached to the token
    # is the version where it first appeared.
    tokenlist = [tokenize_annotated(doc, version)
                 for doc, version in doclist]
    cur_tokens = tokenlist[0]
    for tokens in tokenlist[1:]:
        html_annotate_merge_annotations(cur_tokens, tokens)
        cur_tokens = tokens

    # After we've tracked all the tokens, we can combine spans of text
    # that are adjacent and have the same annotation
    cur_tokens = compress_tokens(cur_tokens)
    # And finally add markup
    result = markup_serialize_tokens(cur_tokens, markup)
    return ''.join(result).strip()

def tokenize_annotated(doc, annotation): 
    """Tokenize a document and add an annotation attribute to each token
    """
    tokens = tokenize(doc, include_hrefs=False)
    for tok in tokens: 
        tok.annotation = annotation
    return tokens

def html_annotate_merge_annotations(tokens_old, tokens_new): 
    """Merge the annotations from tokens_old into tokens_new, when the
    tokens in the new document already existed in the old document.
    """
    s = InsensitiveSequenceMatcher(a=tokens_old, b=tokens_new)
    commands = s.get_opcodes()

    for command, i1, i2, j1, j2 in commands:
        if command == 'equal': 
            eq_old = tokens_old[i1:i2]
            eq_new = tokens_new[j1:j2]
            copy_annotations(eq_old, eq_new)

def copy_annotations(src, dest): 
    """
    Copy annotations from the tokens listed in src to the tokens in dest
    """
    assert len(src) == len(dest)
    for src_tok, dest_tok in zip(src, dest): 
        dest_tok.annotation = src_tok.annotation

def compress_tokens(tokens):
    """
    Combine adjacent tokens when there is no HTML between the tokens, 
    and they share an annotation
    """
    result = [tokens[0]] 
    for tok in tokens[1:]: 
        if (not result[-1].post_tags and 
            not tok.pre_tags and 
            result[-1].annotation == tok.annotation): 
            compress_merge_back(result, tok)
        else: 
            result.append(tok)
    return result

def compress_merge_back(tokens, tok): 
    """ Merge tok into the last element of tokens (modifying the list of
    tokens in-place).  """
    last = tokens[-1]
    if type(last) is not token or type(tok) is not token: 
        tokens.append(tok)
    else:
        text = _unicode(last)
        if last.trailing_whitespace:
            text += last.trailing_whitespace
        text += tok
        merged = token(text,
                       pre_tags=last.pre_tags,
                       post_tags=tok.post_tags,
                       trailing_whitespace=tok.trailing_whitespace)
        merged.annotation = last.annotation
        tokens[-1] = merged
    
def markup_serialize_tokens(tokens, markup_func):
    """
    Serialize the list of tokens into a list of text chunks, calling
    markup_func around text to add annotations.
    """
    for token in tokens:
        for pre in token.pre_tags:
            yield pre
        html = token.html()
        html = markup_func(html, token.annotation)
        if token.trailing_whitespace:
            html += token.trailing_whitespace
        yield html
        for post in token.post_tags:
            yield post


############################################################
## HTML Diffs
############################################################

def htmldiff(old_html, new_html):
    ## FIXME: this should take parsed documents too, and use their body
    ## or other content.
    """ Do a diff of the old and new document.  The documents are HTML
    *fragments* (str/UTF8 or unicode), they are not complete documents
    (i.e., no <html> tag).

    Returns HTML with <ins> and <del> tags added around the
    appropriate text.  

    Markup is generally ignored, with the markup from new_html
    preserved, and possibly some markup from old_html (though it is
    considered acceptable to lose some of the old markup).  Only the
    words in the HTML are diffed.  The exception is <img> tags, which
    are treated like words, and the href attribute of <a> tags, which
    are noted inside the tag itself when there are changes.
    """ 
    old_html_tokens = tokenize(old_html)
    new_html_tokens = tokenize(new_html)
    result = htmldiff_tokens(old_html_tokens, new_html_tokens)
    result = ''.join(result).strip()
    return fixup_ins_del_tags(result)

def htmldiff_tokens(html1_tokens, html2_tokens):
    """ Does a diff on the tokens themselves, returning a list of text
    chunks (not tokens).
    """
    # There are several passes as we do the differences.  The tokens
    # isolate the portion of the content we care to diff; difflib does
    # all the actual hard work at that point.  
    #
    # Then we must create a valid document from pieces of both the old
    # document and the new document.  We generally prefer to take
    # markup from the new document, and only do a best effort attempt
    # to keep markup from the old document; anything that we can't
    # resolve we throw away.  Also we try to put the deletes as close
    # to the location where we think they would have been -- because
    # we are only keeping the markup from the new document, it can be
    # fuzzy where in the new document the old text would have gone.
    # Again we just do a best effort attempt.
    s = InsensitiveSequenceMatcher(a=html1_tokens, b=html2_tokens)
    commands = s.get_opcodes()
    result = []
    for command, i1, i2, j1, j2 in commands:
        if command == 'equal':
            result.extend(expand_tokens(html2_tokens[j1:j2], equal=True))
            continue
        if command == 'insert' or command == 'replace':
            ins_tokens = expand_tokens(html2_tokens[j1:j2])
            merge_insert(ins_tokens, result)
        if command == 'delete' or command == 'replace':
            del_tokens = expand_tokens(html1_tokens[i1:i2])
            merge_delete(del_tokens, result)
    # If deletes were inserted directly as <del> then we'd have an
    # invalid document at this point.  Instead we put in special
    # markers, and when the complete diffed document has been created
    # we try to move the deletes around and resolve any problems.
    result = cleanup_delete(result)

    return result

def expand_tokens(tokens, equal=False):
    """Given a list of tokens, return a generator of the chunks of
    text for the data in the tokens.
    """
    for token in tokens:
        for pre in token.pre_tags:
            yield pre
        if not equal or not token.hide_when_equal:
            if token.trailing_whitespace:
                yield token.html() + token.trailing_whitespace
            else:
                yield token.html()
        for post in token.post_tags:
            yield post

def merge_insert(ins_chunks, doc):
    """ doc is the already-handled document (as a list of text chunks);
    here we add <ins>ins_chunks</ins> to the end of that.  """
    # Though we don't throw away unbalanced_start or unbalanced_end
    # (we assume there is accompanying markup later or earlier in the
    # document), we only put <ins> around the balanced portion.
    unbalanced_start, balanced, unbalanced_end = split_unbalanced(ins_chunks)
    doc.extend(unbalanced_start)
    if doc and not doc[-1].endswith(' '):
        # Fix up the case where the word before the insert didn't end with 
        # a space
        doc[-1] += ' '
    doc.append('<ins>')
    if balanced and balanced[-1].endswith(' '):
        # We move space outside of </ins>
        balanced[-1] = balanced[-1][:-1]
    doc.extend(balanced)
    doc.append('</ins> ')
    doc.extend(unbalanced_end)

# These are sentinels to represent the start and end of a <del>
# segment, until we do the cleanup phase to turn them into proper
# markup:
class DEL_START:
    pass
class DEL_END:
    pass

class NoDeletes(Exception):
    """ Raised when the document no longer contains any pending deletes
    (DEL_START/DEL_END) """

def merge_delete(del_chunks, doc):
    """ Adds the text chunks in del_chunks to the document doc (another
    list of text chunks) with marker to show it is a delete.
    cleanup_delete later resolves these markers into <del> tags."""
    doc.append(DEL_START)
    doc.extend(del_chunks)
    doc.append(DEL_END)

def cleanup_delete(chunks):
    """ Cleans up any DEL_START/DEL_END markers in the document, replacing
    them with <del></del>.  To do this while keeping the document
    valid, it may need to drop some tags (either start or end tags).

    It may also move the del into adjacent tags to try to move it to a
    similar location where it was originally located (e.g., moving a
    delete into preceding <div> tag, if the del looks like (DEL_START,
    'Text</div>', DEL_END)"""
    while 1:
        # Find a pending DEL_START/DEL_END, splitting the document
        # into stuff-preceding-DEL_START, stuff-inside, and
        # stuff-following-DEL_END
        try:
            pre_delete, delete, post_delete = split_delete(chunks)
        except NoDeletes:
            # Nothing found, we've cleaned up the entire doc
            break
        # The stuff-inside-DEL_START/END may not be well balanced
        # markup.  First we figure out what unbalanced portions there are:
        unbalanced_start, balanced, unbalanced_end = split_unbalanced(delete)
        # Then we move the span forward and/or backward based on these
        # unbalanced portions:
        locate_unbalanced_start(unbalanced_start, pre_delete, post_delete)
        locate_unbalanced_end(unbalanced_end, pre_delete, post_delete)
        doc = pre_delete
        if doc and not doc[-1].endswith(' '):
            # Fix up case where the word before us didn't have a trailing space
            doc[-1] += ' '
        doc.append('<del>')
        if balanced and balanced[-1].endswith(' '):
            # We move space outside of </del>
            balanced[-1] = balanced[-1][:-1]
        doc.extend(balanced)
        doc.append('</del> ')
        doc.extend(post_delete)
        chunks = doc
    return chunks

def split_unbalanced(chunks):
    """Return (unbalanced_start, balanced, unbalanced_end), where each is
    a list of text and tag chunks.

    unbalanced_start is a list of all the tags that are opened, but
    not closed in this span.  Similarly, unbalanced_end is a list of
    tags that are closed but were not opened.  Extracting these might
    mean some reordering of the chunks."""
    start = []
    end = []
    tag_stack = []
    balanced = []
    for chunk in chunks:
        if not chunk.startswith('<'):
            balanced.append(chunk)
            continue
        endtag = chunk[1] == '/'
        name = chunk.split()[0].strip('<>/')
        if name in empty_tags:
            balanced.append(chunk)
            continue
        if endtag:
            if tag_stack and tag_stack[-1][0] == name:
                balanced.append(chunk)
                name, pos, tag = tag_stack.pop()
                balanced[pos] = tag
            elif tag_stack:
                start.extend([tag for name, pos, tag in tag_stack])
                tag_stack = []
                end.append(chunk)
            else:
                end.append(chunk)
        else:
            tag_stack.append((name, len(balanced), chunk))
            balanced.append(None)
    start.extend(
        [chunk for name, pos, chunk in tag_stack])
    balanced = [chunk for chunk in balanced if chunk is not None]
    return start, balanced, end

def split_delete(chunks):
    """ Returns (stuff_before_DEL_START, stuff_inside_DEL_START_END,
    stuff_after_DEL_END).  Returns the first case found (there may be
    more DEL_STARTs in stuff_after_DEL_END).  Raises NoDeletes if
    there's no DEL_START found. """
    try:
        pos = chunks.index(DEL_START)
    except ValueError:
        raise NoDeletes
    pos2 = chunks.index(DEL_END)
    return chunks[:pos], chunks[pos+1:pos2], chunks[pos2+1:]

def locate_unbalanced_start(unbalanced_start, pre_delete, post_delete):
    """ pre_delete and post_delete implicitly point to a place in the
    document (where the two were split).  This moves that point (by
    popping items from one and pushing them onto the other).  It moves
    the point to try to find a place where unbalanced_start applies.

    As an example::

        >>> unbalanced_start = ['<div>']
        >>> doc = ['<p>', 'Text', '</p>', '<div>', 'More Text', '</div>']
        >>> pre, post = doc[:3], doc[3:]
        >>> pre, post
        (['<p>', 'Text', '</p>'], ['<div>', 'More Text', '</div>'])
        >>> locate_unbalanced_start(unbalanced_start, pre, post)
        >>> pre, post
        (['<p>', 'Text', '</p>', '<div>'], ['More Text', '</div>'])

    As you can see, we moved the point so that the dangling <div> that
    we found will be effectively replaced by the div in the original
    document.  If this doesn't work out, we just throw away
    unbalanced_start without doing anything.
    """
    while 1:
        if not unbalanced_start:
            # We have totally succeeded in finding the position
            break
        finding = unbalanced_start[0]
        finding_name = finding.split()[0].strip('<>')
        if not post_delete:
            break
        next = post_delete[0]
        if next is DEL_START or not next.startswith('<'):
            # Reached a word, we can't move the delete text forward
            break
        if next[1] == '/':
            # Reached a closing tag, can we go further?  Maybe not...
            break
        name = next.split()[0].strip('<>')
        if name == 'ins':
            # Can't move into an insert
            break
        assert name != 'del', (
            "Unexpected delete tag: %r" % next)
        if name == finding_name:
            unbalanced_start.pop(0)
            pre_delete.append(post_delete.pop(0))
        else:
            # Found a tag that doesn't match
            break

def locate_unbalanced_end(unbalanced_end, pre_delete, post_delete):
    """ like locate_unbalanced_start, except handling end tags and
    possibly moving the point earlier in the document.  """
    while 1:
        if not unbalanced_end:
            # Success
            break
        finding = unbalanced_end[-1]
        finding_name = finding.split()[0].strip('<>/')
        if not pre_delete:
            break
        next = pre_delete[-1]
        if next is DEL_END or not next.startswith('</'):
            # A word or a start tag
            break
        name = next.split()[0].strip('<>/')
        if name == 'ins' or name == 'del':
            # Can't move into an insert or delete
            break
        if name == finding_name:
            unbalanced_end.pop()
            post_delete.insert(0, pre_delete.pop())
        else:
            # Found a tag that doesn't match
            break

class token(_unicode):
    """ Represents a diffable token, generally a word that is displayed to
    the user.  Opening tags are attached to this token when they are
    adjacent (pre_tags) and closing tags that follow the word
    (post_tags).  Some exceptions occur when there are empty tags
    adjacent to a word, so there may be close tags in pre_tags, or
    open tags in post_tags.

    We also keep track of whether the word was originally followed by
    whitespace, even though we do not want to treat the word as
    equivalent to a similar word that does not have a trailing
    space."""

    # When this is true, the token will be eliminated from the
    # displayed diff if no change has occurred:
    hide_when_equal = False

    def __new__(cls, text, pre_tags=None, post_tags=None, trailing_whitespace=""):
        obj = _unicode.__new__(cls, text)

        if pre_tags is not None:
            obj.pre_tags = pre_tags
        else:
            obj.pre_tags = []

        if post_tags is not None:
            obj.post_tags = post_tags
        else:
            obj.post_tags = []

        obj.trailing_whitespace = trailing_whitespace

        return obj

    def __repr__(self):
        return 'token(%s, %r, %r, %r)' % (_unicode.__repr__(self), self.pre_tags,
                                          self.post_tags, self.trailing_whitespace)

    def html(self):
        return _unicode(self)

class tag_token(token):

    """ Represents a token that is actually a tag.  Currently this is just
    the <img> tag, which takes up visible space just like a word but
    is only represented in a document by a tag.  """

    def __new__(cls, tag, data, html_repr, pre_tags=None, 
                post_tags=None, trailing_whitespace=""):
        obj = token.__new__(cls, "%s: %s" % (type, data), 
                            pre_tags=pre_tags, 
                            post_tags=post_tags, 
                            trailing_whitespace=trailing_whitespace)
        obj.tag = tag
        obj.data = data
        obj.html_repr = html_repr
        return obj

    def __repr__(self):
        return 'tag_token(%s, %s, html_repr=%s, post_tags=%r, pre_tags=%r, trailing_whitespace=%r)' % (
            self.tag, 
            self.data, 
            self.html_repr, 
            self.pre_tags, 
            self.post_tags, 
            self.trailing_whitespace)
    def html(self):
        return self.html_repr

class href_token(token):

    """ Represents the href in an anchor tag.  Unlike other words, we only
    show the href when it changes.  """

    hide_when_equal = True

    def html(self):
        return ' Link: %s' % self

def tokenize(html, include_hrefs=True):
    """
    Parse the given HTML and returns token objects (words with attached tags).

    This parses only the content of a page; anything in the head is
    ignored, and the <head> and <body> elements are themselves
    optional.  The content is then parsed by lxml, which ensures the
    validity of the resulting parsed document (though lxml may make
    incorrect guesses when the markup is particular bad).

    <ins> and <del> tags are also eliminated from the document, as
    that gets confusing.

    If include_hrefs is true, then the href attribute of <a> tags is
    included as a special kind of diffable token."""
    if etree.iselement(html):
        body_el = html
    else:
        body_el = parse_html(html, cleanup=True)
    # Then we split the document into text chunks for each tag, word, and end tag:
    chunks = flatten_el(body_el, skip_tag=True, include_hrefs=include_hrefs)
    # Finally re-joining them into token objects:
    return fixup_chunks(chunks)

def parse_html(html, cleanup=True):
    """
    Parses an HTML fragment, returning an lxml element.  Note that the HTML will be
    wrapped in a <div> tag that was not in the original document.

    If cleanup is true, make sure there's no <head> or <body>, and get
    rid of any <ins> and <del> tags.
    """
    if cleanup:
        # This removes any extra markup or structure like <head>:
        html = cleanup_html(html)
    return fragment_fromstring(html, create_parent=True)

_body_re = re.compile(r'<body.*?>', re.I|re.S)
_end_body_re = re.compile(r'</body.*?>', re.I|re.S)
_ins_del_re = re.compile(r'</?(ins|del).*?>', re.I|re.S)

def cleanup_html(html):
    """ This 'cleans' the HTML, meaning that any page structure is removed
    (only the contents of <body> are used, if there is any <body).
    Also <ins> and <del> tags are removed.  """
    match = _body_re.search(html)
    if match:
        html = html[match.end():]
    match = _end_body_re.search(html)
    if match:
        html = html[:match.start()]
    html = _ins_del_re.sub('', html)
    return html
    

end_whitespace_re = re.compile(r'[ \t\n\r]$')

def split_trailing_whitespace(word):
    """
    This function takes a word, such as 'test\n\n' and returns ('test','\n\n')
    """
    stripped_length = len(word.rstrip())
    return word[0:stripped_length], word[stripped_length:]


def fixup_chunks(chunks):
    """
    This function takes a list of chunks and produces a list of tokens.
    """
    tag_accum = []
    cur_word = None
    result = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            if chunk[0] == 'img':
                src = chunk[1]
                tag, trailing_whitespace = split_trailing_whitespace(chunk[2])
                cur_word = tag_token('img', src, html_repr=tag,
                                     pre_tags=tag_accum,
                                     trailing_whitespace=trailing_whitespace)
                tag_accum = []
                result.append(cur_word)

            elif chunk[0] == 'href':
                href = chunk[1]
                cur_word = href_token(href, pre_tags=tag_accum, trailing_whitespace=" ")
                tag_accum = []
                result.append(cur_word)
            continue

        if is_word(chunk):
            chunk, trailing_whitespace = split_trailing_whitespace(chunk)
            cur_word = token(chunk, pre_tags=tag_accum, trailing_whitespace=trailing_whitespace)
            tag_accum = []
            result.append(cur_word)

        elif is_start_tag(chunk):
            tag_accum.append(chunk)

        elif is_end_tag(chunk):
            if tag_accum:
                tag_accum.append(chunk)
            else:
                assert cur_word, (
                    "Weird state, cur_word=%r, result=%r, chunks=%r of %r"
                    % (cur_word, result, chunk, chunks))
                cur_word.post_tags.append(chunk)
        else:
            assert False

    if not result:
        return [token('', pre_tags=tag_accum)]
    else:
        result[-1].post_tags.extend(tag_accum)

    return result


# All the tags in HTML that don't require end tags:
empty_tags = (
    'param', 'img', 'area', 'br', 'basefont', 'input',
    'base', 'meta', 'link', 'col')

block_level_tags = (
    'address',
    'blockquote',
    'center',
    'dir',
    'div',
    'dl',
    'fieldset',
    'form',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'hr',
    'isindex',
    'menu',
    'noframes',
    'noscript',
    'ol',
    'p',
    'pre',
    'table',
    'ul',
    )

block_level_container_tags = (
    'dd',
    'dt',
    'frameset',
    'li',
    'tbody',
    'td',
    'tfoot',
    'th',
    'thead',
    'tr',
    )


def flatten_el(el, include_hrefs, skip_tag=False):
    """ Takes an lxml element el, and generates all the text chunks for
    that tag.  Each start tag is a chunk, each word is a chunk, and each
    end tag is a chunk.

    If skip_tag is true, then the outermost container tag is
    not returned (just its contents)."""
    if not skip_tag:
        if el.tag == 'img':
            yield ('img', el.get('src'), start_tag(el))
        else:
            yield start_tag(el)
    if el.tag in empty_tags and not el.text and not len(el) and not el.tail:
        return
    start_words = split_words(el.text)
    for word in start_words:
        yield html_escape(word)
    for child in el:
        for item in flatten_el(child, include_hrefs=include_hrefs):
            yield item
    if el.tag == 'a' and el.get('href') and include_hrefs:
        yield ('href', el.get('href'))
    if not skip_tag:
        yield end_tag(el)
        end_words = split_words(el.tail)
        for word in end_words:
            yield html_escape(word)

split_words_re = re.compile(r'\S+(?:\s+|$)', re.U)

def split_words(text):
    """ Splits some text into words. Includes trailing whitespace
    on each word when appropriate.  """
    if not text or not text.strip():
        return []

    words = split_words_re.findall(text)
    return words

start_whitespace_re = re.compile(r'^[ \t\n\r]')

def start_tag(el):
    """
    The text representation of the start tag for a tag.
    """
    return '<%s%s>' % (
        el.tag, ''.join([' %s="%s"' % (name, html_escape(value, True))
                         for name, value in el.attrib.items()]))

def end_tag(el):
    """ The text representation of an end tag for a tag.  Includes
    trailing whitespace when appropriate.  """
    if el.tail and start_whitespace_re.search(el.tail):
        extra = ' '
    else:
        extra = ''
    return '</%s>%s' % (el.tag, extra)

def is_word(tok):
    return not tok.startswith('<')

def is_end_tag(tok):
    return tok.startswith('</')

def is_start_tag(tok):
    return tok.startswith('<') and not tok.startswith('</')

def fixup_ins_del_tags(html):
    """ Given an html string, move any <ins> or <del> tags inside of any
    block-level elements, e.g. transform <ins><p>word</p></ins> to
    <p><ins>word</ins></p> """
    doc = parse_html(html, cleanup=False)
    _fixup_ins_del_tags(doc)
    html = serialize_html_fragment(doc, skip_outer=True)
    return html

def serialize_html_fragment(el, skip_outer=False):
    """ Serialize a single lxml element as HTML.  The serialized form
    includes the elements tail.  

    If skip_outer is true, then don't serialize the outermost tag
    """
    assert not isinstance(el, basestring), (
        "You should pass in an element, not a string like %r" % el)
    html = etree.tostring(el, method="html", encoding=_unicode)
    if skip_outer:
        # Get rid of the extra starting tag:
        html = html[html.find('>')+1:]
        # Get rid of the extra end tag:
        html = html[:html.rfind('<')]
        return html.strip()
    else:
        return html

def _fixup_ins_del_tags(doc):
    """fixup_ins_del_tags that works on an lxml document in-place
    """
    for tag in ['ins', 'del']:
        for el in doc.xpath('descendant-or-self::%s' % tag):
            if not _contains_block_level_tag(el):
                continue
            _move_el_inside_block(el, tag=tag)
            el.drop_tag()
            #_merge_element_contents(el)

def _contains_block_level_tag(el):
    """True if the element contains any block-level elements, like <p>, <td>, etc.
    """
    if el.tag in block_level_tags or el.tag in block_level_container_tags:
        return True
    for child in el:
        if _contains_block_level_tag(child):
            return True
    return False

def _move_el_inside_block(el, tag):
    """ helper for _fixup_ins_del_tags; actually takes the <ins> etc tags
    and moves them inside any block-level tags.  """
    for child in el:
        if _contains_block_level_tag(child):
            break
    else:
        # No block-level tags in any child
        children_tag = etree.Element(tag)
        children_tag.text = el.text
        el.text = None
        children_tag.extend(list(el))
        el[:] = [children_tag]
        return
    for child in list(el):
        if _contains_block_level_tag(child):
            _move_el_inside_block(child, tag)
            if child.tail:
                tail_tag = etree.Element(tag)
                tail_tag.text = child.tail
                child.tail = None
                el.insert(el.index(child)+1, tail_tag)
        else:
            child_tag = etree.Element(tag)
            el.replace(child, child_tag)
            child_tag.append(child)
    if el.text:
        text_tag = etree.Element(tag)
        text_tag.text = el.text
        el.text = None
        el.insert(0, text_tag)
            
def _merge_element_contents(el):
    """
    Removes an element, but merges its contents into its place, e.g.,
    given <p>Hi <i>there!</i></p>, if you remove the <i> element you get
    <p>Hi there!</p>
    """
    parent = el.getparent()
    text = el.text or ''
    if el.tail:
        if not len(el):
            text += el.tail
        else:
            if el[-1].tail:
                el[-1].tail += el.tail
            else:
                el[-1].tail = el.tail
    index = parent.index(el)
    if text:
        if index == 0:
            previous = None
        else:
            previous = parent[index-1]
        if previous is None:
            if parent.text:
                parent.text += text
            else:
                parent.text = text
        else:
            if previous.tail:
                previous.tail += text
            else:
                previous.tail = text
    parent[index:index+1] = el.getchildren()

class InsensitiveSequenceMatcher(difflib.SequenceMatcher):
    """
    Acts like SequenceMatcher, but tries not to find very small equal
    blocks amidst large spans of changes
    """

    threshold = 2
    
    def get_matching_blocks(self):
        size = min(len(self.b), len(self.b))
        threshold = min(self.threshold, size / 4)
        actual = difflib.SequenceMatcher.get_matching_blocks(self)
        return [item for item in actual
                if item[2] > threshold
                or not item[2]]

if __name__ == '__main__':
    from lxml.html import _diffcommand
    _diffcommand.main()
    
