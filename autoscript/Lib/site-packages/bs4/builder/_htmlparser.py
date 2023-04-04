# encoding: utf-8
"""Use the HTMLParser library to parse HTML files that aren't too bad."""

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    'HTMLParserTreeBuilder',
    ]

from html.parser import HTMLParser

import sys
import warnings

from bs4.element import (
    CData,
    Comment,
    Declaration,
    Doctype,
    ProcessingInstruction,
    )
from bs4.dammit import EntitySubstitution, UnicodeDammit

from bs4.builder import (
    DetectsXMLParsedAsHTML,
    ParserRejectedMarkup,
    HTML,
    HTMLTreeBuilder,
    STRICT,
    )


HTMLPARSER = 'html.parser'

class BeautifulSoupHTMLParser(HTMLParser, DetectsXMLParsedAsHTML):
    """A subclass of the Python standard library's HTMLParser class, which
    listens for HTMLParser events and translates them into calls
    to Beautiful Soup's tree construction API.
    """

    # Strategies for handling duplicate attributes
    IGNORE = 'ignore'
    REPLACE = 'replace'
    
    def __init__(self, *args, **kwargs):
        """Constructor.

        :param on_duplicate_attribute: A strategy for what to do if a
            tag includes the same attribute more than once. Accepted
            values are: REPLACE (replace earlier values with later
            ones, the default), IGNORE (keep the earliest value
            encountered), or a callable. A callable must take three
            arguments: the dictionary of attributes already processed,
            the name of the duplicate attribute, and the most recent value
            encountered.           
        """
        self.on_duplicate_attribute = kwargs.pop(
            'on_duplicate_attribute', self.REPLACE
        )
        HTMLParser.__init__(self, *args, **kwargs)

        # Keep a list of empty-element tags that were encountered
        # without an explicit closing tag. If we encounter a closing tag
        # of this type, we'll associate it with one of those entries.
        #
        # This isn't a stack because we don't care about the
        # order. It's a list of closing tags we've already handled and
        # will ignore, assuming they ever show up.
        self.already_closed_empty_element = []

        self._initialize_xml_detector()

    def error(self, message):
        # NOTE: This method is required so long as Python 3.9 is
        # supported. The corresponding code is removed from HTMLParser
        # in 3.5, but not removed from ParserBase until 3.10.
        # https://github.com/python/cpython/issues/76025
        #
        # The original implementation turned the error into a warning,
        # but in every case I discovered, this made HTMLParser
        # immediately crash with an error message that was less
        # helpful than the warning. The new implementation makes it
        # more clear that html.parser just can't parse this
        # markup. The 3.10 implementation does the same, though it
        # raises AssertionError rather than calling a method. (We
        # catch this error and wrap it in a ParserRejectedMarkup.)
        raise ParserRejectedMarkup(message)

    def handle_startendtag(self, name, attrs):
        """Handle an incoming empty-element tag.

        This is only called when the markup looks like <tag/>.

        :param name: Name of the tag.
        :param attrs: Dictionary of the tag's attributes.
        """
        # is_startend() tells handle_starttag not to close the tag
        # just because its name matches a known empty-element tag. We
        # know that this is an empty-element tag and we want to call
        # handle_endtag ourselves.
        tag = self.handle_starttag(name, attrs, handle_empty_element=False)
        self.handle_endtag(name)
        
    def handle_starttag(self, name, attrs, handle_empty_element=True):
        """Handle an opening tag, e.g. '<tag>'

        :param name: Name of the tag.
        :param attrs: Dictionary of the tag's attributes.
        :param handle_empty_element: True if this tag is known to be
            an empty-element tag (i.e. there is not expected to be any
            closing tag).
        """
        # XXX namespace
        attr_dict = {}
        for key, value in attrs:
            # Change None attribute values to the empty string
            # for consistency with the other tree builders.
            if value is None:
                value = ''
            if key in attr_dict:
                # A single attribute shows up multiple times in this
                # tag. How to handle it depends on the
                # on_duplicate_attribute setting.
                on_dupe = self.on_duplicate_attribute
                if on_dupe == self.IGNORE:
                    pass
                elif on_dupe in (None, self.REPLACE):
                    attr_dict[key] = value
                else:
                    on_dupe(attr_dict, key, value)
            else:
                attr_dict[key] = value
            attrvalue = '""'
        #print("START", name)
        sourceline, sourcepos = self.getpos()
        tag = self.soup.handle_starttag(
            name, None, None, attr_dict, sourceline=sourceline,
            sourcepos=sourcepos
        )
        if tag and tag.is_empty_element and handle_empty_element:
            # Unlike other parsers, html.parser doesn't send separate end tag
            # events for empty-element tags. (It's handled in
            # handle_startendtag, but only if the original markup looked like
            # <tag/>.)
            #
            # So we need to call handle_endtag() ourselves. Since we
            # know the start event is identical to the end event, we
            # don't want handle_endtag() to cross off any previous end
            # events for tags of this name.
            self.handle_endtag(name, check_already_closed=False)

            # But we might encounter an explicit closing tag for this tag
            # later on. If so, we want to ignore it.
            self.already_closed_empty_element.append(name)

        if self._root_tag is None:
            self._root_tag_encountered(name)
            
    def handle_endtag(self, name, check_already_closed=True):
        """Handle a closing tag, e.g. '</tag>'
        
        :param name: A tag name.
        :param check_already_closed: True if this tag is expected to
           be the closing portion of an empty-element tag,
           e.g. '<tag></tag>'.
        """
        #print("END", name)
        if check_already_closed and name in self.already_closed_empty_element:
            # This is a redundant end tag for an empty-element tag.
            # We've already called handle_endtag() for it, so just
            # check it off the list.
            #print("ALREADY CLOSED", name)
            self.already_closed_empty_element.remove(name)
        else:
            self.soup.handle_endtag(name)
            
    def handle_data(self, data):
        """Handle some textual data that shows up between tags."""
        self.soup.handle_data(data)

    def handle_charref(self, name):
        """Handle a numeric character reference by converting it to the
        corresponding Unicode character and treating it as textual
        data.

        :param name: Character number, possibly in hexadecimal.
        """
        # TODO: This was originally a workaround for a bug in
        # HTMLParser. (http://bugs.python.org/issue13633) The bug has
        # been fixed, but removing this code still makes some
        # Beautiful Soup tests fail. This needs investigation.
        if name.startswith('x'):
            real_name = int(name.lstrip('x'), 16)
        elif name.startswith('X'):
            real_name = int(name.lstrip('X'), 16)
        else:
            real_name = int(name)

        data = None
        if real_name < 256:
            # HTML numeric entities are supposed to reference Unicode
            # code points, but sometimes they reference code points in
            # some other encoding (ahem, Windows-1252). E.g. &#147;
            # instead of &#201; for LEFT DOUBLE QUOTATION MARK. This
            # code tries to detect this situation and compensate.
            for encoding in (self.soup.original_encoding, 'windows-1252'):
                if not encoding:
                    continue
                try:
                    data = bytearray([real_name]).decode(encoding)
                except UnicodeDecodeError as e:
                    pass
        if not data:
            try:
                data = chr(real_name)
            except (ValueError, OverflowError) as e:
                pass
        data = data or "\N{REPLACEMENT CHARACTER}"
        self.handle_data(data)

    def handle_entityref(self, name):
        """Handle a named entity reference by converting it to the
        corresponding Unicode character(s) and treating it as textual
        data.

        :param name: Name of the entity reference.
        """
        character = EntitySubstitution.HTML_ENTITY_TO_CHARACTER.get(name)
        if character is not None:
            data = character
        else:
            # If this were XML, it would be ambiguous whether "&foo"
            # was an character entity reference with a missing
            # semicolon or the literal string "&foo". Since this is
            # HTML, we have a complete list of all character entity references,
            # and this one wasn't found, so assume it's the literal string "&foo".
            data = "&%s" % name
        self.handle_data(data)

    def handle_comment(self, data):
        """Handle an HTML comment.

        :param data: The text of the comment.
        """
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(Comment)

    def handle_decl(self, data):
        """Handle a DOCTYPE declaration.

        :param data: The text of the declaration.
        """
        self.soup.endData()
        data = data[len("DOCTYPE "):]
        self.soup.handle_data(data)
        self.soup.endData(Doctype)

    def unknown_decl(self, data):
        """Handle a declaration of unknown type -- probably a CDATA block.

        :param data: The text of the declaration.
        """
        if data.upper().startswith('CDATA['):
            cls = CData
            data = data[len('CDATA['):]
        else:
            cls = Declaration
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(cls)

    def handle_pi(self, data):
        """Handle a processing instruction.

        :param data: The text of the instruction.
        """
        self.soup.endData()
        self.soup.handle_data(data)
        self._document_might_be_xml(data)
        self.soup.endData(ProcessingInstruction)


class HTMLParserTreeBuilder(HTMLTreeBuilder):
    """A Beautiful soup `TreeBuilder` that uses the `HTMLParser` parser,
    found in the Python standard library.
    """
    is_xml = False
    picklable = True
    NAME = HTMLPARSER
    features = [NAME, HTML, STRICT]

    # The html.parser knows which line number and position in the
    # original file is the source of an element.
    TRACKS_LINE_NUMBERS = True

    def __init__(self, parser_args=None, parser_kwargs=None, **kwargs):
        """Constructor.

        :param parser_args: Positional arguments to pass into 
            the BeautifulSoupHTMLParser constructor, once it's
            invoked.
        :param parser_kwargs: Keyword arguments to pass into 
            the BeautifulSoupHTMLParser constructor, once it's
            invoked.
        :param kwargs: Keyword arguments for the superclass constructor.
        """
        # Some keyword arguments will be pulled out of kwargs and placed
        # into parser_kwargs.
        extra_parser_kwargs = dict()
        for arg in ('on_duplicate_attribute',):
            if arg in kwargs:
                value = kwargs.pop(arg)
                extra_parser_kwargs[arg] = value
        super(HTMLParserTreeBuilder, self).__init__(**kwargs)
        parser_args = parser_args or []
        parser_kwargs = parser_kwargs or {}
        parser_kwargs.update(extra_parser_kwargs)
        parser_kwargs['convert_charrefs'] = False
        self.parser_args = (parser_args, parser_kwargs)
        
    def prepare_markup(self, markup, user_specified_encoding=None,
                       document_declared_encoding=None, exclude_encodings=None):

        """Run any preliminary steps necessary to make incoming markup
        acceptable to the parser.

        :param markup: Some markup -- probably a bytestring.
        :param user_specified_encoding: The user asked to try this encoding.
        :param document_declared_encoding: The markup itself claims to be
            in this encoding.
        :param exclude_encodings: The user asked _not_ to try any of
            these encodings.

        :yield: A series of 4-tuples:
         (markup, encoding, declared encoding,
          has undergone character replacement)

         Each 4-tuple represents a strategy for converting the
         document to Unicode and parsing it. Each strategy will be tried 
         in turn.
        """
        if isinstance(markup, str):
            # Parse Unicode as-is.
            yield (markup, None, None, False)
            return

        # Ask UnicodeDammit to sniff the most likely encoding.

        # This was provided by the end-user; treat it as a known
        # definite encoding per the algorithm laid out in the HTML5
        # spec.  (See the EncodingDetector class for details.)
        known_definite_encodings = [user_specified_encoding]

        # This was found in the document; treat it as a slightly lower-priority
        # user encoding.
        user_encodings = [document_declared_encoding]

        try_encodings = [user_specified_encoding, document_declared_encoding]
        dammit = UnicodeDammit(
            markup,
            known_definite_encodings=known_definite_encodings,
            user_encodings=user_encodings,
            is_html=True,
            exclude_encodings=exclude_encodings
        )
        yield (dammit.markup, dammit.original_encoding,
               dammit.declared_html_encoding,
               dammit.contains_replacement_characters)

    def feed(self, markup):
        """Run some incoming markup through some parsing process,
        populating the `BeautifulSoup` object in self.soup.
        """
        args, kwargs = self.parser_args
        parser = BeautifulSoupHTMLParser(*args, **kwargs)
        parser.soup = self.soup
        try:
            parser.feed(markup)
        except AssertionError as e:
            # html.parser raises AssertionError in rare cases to
            # indicate a fatal problem with the markup, especially
            # when there's an error in the doctype declaration.
            raise ParserRejectedMarkup(e)
        parser.close()
        parser.already_closed_empty_element = []
