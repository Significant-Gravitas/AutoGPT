# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    'LXMLTreeBuilderForXML',
    'LXMLTreeBuilder',
    ]

try:
    from collections.abc import Callable # Python 3.6
except ImportError as e:
    from collections import Callable

from io import BytesIO
from io import StringIO
from lxml import etree
from bs4.element import (
    Comment,
    Doctype,
    NamespacedAttribute,
    ProcessingInstruction,
    XMLProcessingInstruction,
)
from bs4.builder import (
    DetectsXMLParsedAsHTML,
    FAST,
    HTML,
    HTMLTreeBuilder,
    PERMISSIVE,
    ParserRejectedMarkup,
    TreeBuilder,
    XML)
from bs4.dammit import EncodingDetector

LXML = 'lxml'

def _invert(d):
    "Invert a dictionary."
    return dict((v,k) for k, v in list(d.items()))

class LXMLTreeBuilderForXML(TreeBuilder):
    DEFAULT_PARSER_CLASS = etree.XMLParser

    is_xml = True
    processing_instruction_class = XMLProcessingInstruction

    NAME = "lxml-xml"
    ALTERNATE_NAMES = ["xml"]

    # Well, it's permissive by XML parser standards.
    features = [NAME, LXML, XML, FAST, PERMISSIVE]

    CHUNK_SIZE = 512

    # This namespace mapping is specified in the XML Namespace
    # standard.
    DEFAULT_NSMAPS = dict(xml='http://www.w3.org/XML/1998/namespace')

    DEFAULT_NSMAPS_INVERTED = _invert(DEFAULT_NSMAPS)

    # NOTE: If we parsed Element objects and looked at .sourceline,
    # we'd be able to see the line numbers from the original document.
    # But instead we build an XMLParser or HTMLParser object to serve
    # as the target of parse messages, and those messages don't include
    # line numbers.
    # See: https://bugs.launchpad.net/lxml/+bug/1846906
    
    def initialize_soup(self, soup):
        """Let the BeautifulSoup object know about the standard namespace
        mapping.

        :param soup: A `BeautifulSoup`.
        """
        super(LXMLTreeBuilderForXML, self).initialize_soup(soup)
        self._register_namespaces(self.DEFAULT_NSMAPS)

    def _register_namespaces(self, mapping):
        """Let the BeautifulSoup object know about namespaces encountered
        while parsing the document.

        This might be useful later on when creating CSS selectors.

        This will track (almost) all namespaces, even ones that were
        only in scope for part of the document. If two namespaces have
        the same prefix, only the first one encountered will be
        tracked. Un-prefixed namespaces are not tracked.

        :param mapping: A dictionary mapping namespace prefixes to URIs.
        """
        for key, value in list(mapping.items()):
            # This is 'if key' and not 'if key is not None' because we
            # don't track un-prefixed namespaces. Soupselect will
            # treat an un-prefixed namespace as the default, which
            # causes confusion in some cases.
            if key and key not in self.soup._namespaces:
                # Let the BeautifulSoup object know about a new namespace.
                # If there are multiple namespaces defined with the same
                # prefix, the first one in the document takes precedence.
                self.soup._namespaces[key] = value
                
    def default_parser(self, encoding):
        """Find the default parser for the given encoding.

        :param encoding: A string.
        :return: Either a parser object or a class, which
          will be instantiated with default arguments.
        """
        if self._default_parser is not None:
            return self._default_parser
        return etree.XMLParser(
            target=self, strip_cdata=False, recover=True, encoding=encoding)

    def parser_for(self, encoding):
        """Instantiate an appropriate parser for the given encoding.

        :param encoding: A string.
        :return: A parser object such as an `etree.XMLParser`.
        """
        # Use the default parser.
        parser = self.default_parser(encoding)

        if isinstance(parser, Callable):
            # Instantiate the parser with default arguments
            parser = parser(
                target=self, strip_cdata=False, recover=True, encoding=encoding
            )
        return parser

    def __init__(self, parser=None, empty_element_tags=None, **kwargs):
        # TODO: Issue a warning if parser is present but not a
        # callable, since that means there's no way to create new
        # parsers for different encodings.
        self._default_parser = parser
        if empty_element_tags is not None:
            self.empty_element_tags = set(empty_element_tags)
        self.soup = None
        self.nsmaps = [self.DEFAULT_NSMAPS_INVERTED]
        self.active_namespace_prefixes = [dict(self.DEFAULT_NSMAPS)]
        super(LXMLTreeBuilderForXML, self).__init__(**kwargs)
        
    def _getNsTag(self, tag):
        # Split the namespace URL out of a fully-qualified lxml tag
        # name. Copied from lxml's src/lxml/sax.py.
        if tag[0] == '{':
            return tuple(tag[1:].split('}', 1))
        else:
            return (None, tag)

    def prepare_markup(self, markup, user_specified_encoding=None,
                       exclude_encodings=None,
                       document_declared_encoding=None):
        """Run any preliminary steps necessary to make incoming markup
        acceptable to the parser.

        lxml really wants to get a bytestring and convert it to
        Unicode itself. So instead of using UnicodeDammit to convert
        the bytestring to Unicode using different encodings, this
        implementation uses EncodingDetector to iterate over the
        encodings, and tell lxml to try to parse the document as each
        one in turn.

        :param markup: Some markup -- hopefully a bytestring.
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
        is_html = not self.is_xml
        if is_html:
            self.processing_instruction_class = ProcessingInstruction
            # We're in HTML mode, so if we're given XML, that's worth
            # noting.
            DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup)
        else:
            self.processing_instruction_class = XMLProcessingInstruction

        if isinstance(markup, str):
            # We were given Unicode. Maybe lxml can parse Unicode on
            # this system?

            # TODO: This is a workaround for
            # https://bugs.launchpad.net/lxml/+bug/1948551.
            # We can remove it once the upstream issue is fixed.
            if len(markup) > 0 and markup[0] == u'\N{BYTE ORDER MARK}':
                markup = markup[1:]
            yield markup, None, document_declared_encoding, False

        if isinstance(markup, str):
            # No, apparently not. Convert the Unicode to UTF-8 and
            # tell lxml to parse it as UTF-8.
            yield (markup.encode("utf8"), "utf8",
                   document_declared_encoding, False)

        # This was provided by the end-user; treat it as a known
        # definite encoding per the algorithm laid out in the HTML5
        # spec.  (See the EncodingDetector class for details.)
        known_definite_encodings = [user_specified_encoding]

        # This was found in the document; treat it as a slightly lower-priority
        # user encoding.
        user_encodings = [document_declared_encoding]
        detector = EncodingDetector(
            markup, known_definite_encodings=known_definite_encodings,
            user_encodings=user_encodings, is_html=is_html,
            exclude_encodings=exclude_encodings
        )
        for encoding in detector.encodings:
            yield (detector.markup, encoding, document_declared_encoding, False)

    def feed(self, markup):
        if isinstance(markup, bytes):
            markup = BytesIO(markup)
        elif isinstance(markup, str):
            markup = StringIO(markup)

        # Call feed() at least once, even if the markup is empty,
        # or the parser won't be initialized.
        data = markup.read(self.CHUNK_SIZE)
        try:
            self.parser = self.parser_for(self.soup.original_encoding)
            self.parser.feed(data)
            while len(data) != 0:
                # Now call feed() on the rest of the data, chunk by chunk.
                data = markup.read(self.CHUNK_SIZE)
                if len(data) != 0:
                    self.parser.feed(data)
            self.parser.close()
        except (UnicodeDecodeError, LookupError, etree.ParserError) as e:
            raise ParserRejectedMarkup(e)

    def close(self):
        self.nsmaps = [self.DEFAULT_NSMAPS_INVERTED]

    def start(self, name, attrs, nsmap={}):
        # Make sure attrs is a mutable dict--lxml may send an immutable dictproxy.
        attrs = dict(attrs)
        nsprefix = None
        # Invert each namespace map as it comes in.
        if len(nsmap) == 0 and len(self.nsmaps) > 1:
                # There are no new namespaces for this tag, but
                # non-default namespaces are in play, so we need a
                # separate tag stack to know when they end.
                self.nsmaps.append(None)
        elif len(nsmap) > 0:
            # A new namespace mapping has come into play.

            # First, Let the BeautifulSoup object know about it.
            self._register_namespaces(nsmap)

            # Then, add it to our running list of inverted namespace
            # mappings.
            self.nsmaps.append(_invert(nsmap))

            # The currently active namespace prefixes have
            # changed. Calculate the new mapping so it can be stored
            # with all Tag objects created while these prefixes are in
            # scope.
            current_mapping = dict(self.active_namespace_prefixes[-1])
            current_mapping.update(nsmap)

            # We should not track un-prefixed namespaces as we can only hold one
            # and it will be recognized as the default namespace by soupsieve,
            # which may be confusing in some situations.
            if '' in current_mapping:
                del current_mapping['']
            self.active_namespace_prefixes.append(current_mapping)
            
            # Also treat the namespace mapping as a set of attributes on the
            # tag, so we can recreate it later.
            attrs = attrs.copy()
            for prefix, namespace in list(nsmap.items()):
                attribute = NamespacedAttribute(
                    "xmlns", prefix, "http://www.w3.org/2000/xmlns/")
                attrs[attribute] = namespace

        # Namespaces are in play. Find any attributes that came in
        # from lxml with namespaces attached to their names, and
        # turn then into NamespacedAttribute objects.
        new_attrs = {}
        for attr, value in list(attrs.items()):
            namespace, attr = self._getNsTag(attr)
            if namespace is None:
                new_attrs[attr] = value
            else:
                nsprefix = self._prefix_for_namespace(namespace)
                attr = NamespacedAttribute(nsprefix, attr, namespace)
                new_attrs[attr] = value
        attrs = new_attrs

        namespace, name = self._getNsTag(name)
        nsprefix = self._prefix_for_namespace(namespace)
        self.soup.handle_starttag(
            name, namespace, nsprefix, attrs,
            namespaces=self.active_namespace_prefixes[-1]
        )
        
    def _prefix_for_namespace(self, namespace):
        """Find the currently active prefix for the given namespace."""
        if namespace is None:
            return None
        for inverted_nsmap in reversed(self.nsmaps):
            if inverted_nsmap is not None and namespace in inverted_nsmap:
                return inverted_nsmap[namespace]
        return None

    def end(self, name):
        self.soup.endData()
        completed_tag = self.soup.tagStack[-1]
        namespace, name = self._getNsTag(name)
        nsprefix = None
        if namespace is not None:
            for inverted_nsmap in reversed(self.nsmaps):
                if inverted_nsmap is not None and namespace in inverted_nsmap:
                    nsprefix = inverted_nsmap[namespace]
                    break
        self.soup.handle_endtag(name, nsprefix)
        if len(self.nsmaps) > 1:
            # This tag, or one of its parents, introduced a namespace
            # mapping, so pop it off the stack.
            out_of_scope_nsmap = self.nsmaps.pop()

            if out_of_scope_nsmap is not None:
                # This tag introduced a namespace mapping which is no
                # longer in scope. Recalculate the currently active
                # namespace prefixes.
                self.active_namespace_prefixes.pop()
            
    def pi(self, target, data):
        self.soup.endData()
        data = target + ' ' + data
        self.soup.handle_data(data)
        self.soup.endData(self.processing_instruction_class)
        
    def data(self, content):
        self.soup.handle_data(content)

    def doctype(self, name, pubid, system):
        self.soup.endData()
        doctype = Doctype.for_name_and_ids(name, pubid, system)
        self.soup.object_was_parsed(doctype)

    def comment(self, content):
        "Handle comments as Comment objects."
        self.soup.endData()
        self.soup.handle_data(content)
        self.soup.endData(Comment)

    def test_fragment_to_document(self, fragment):
        """See `TreeBuilder`."""
        return '<?xml version="1.0" encoding="utf-8"?>\n%s' % fragment


class LXMLTreeBuilder(HTMLTreeBuilder, LXMLTreeBuilderForXML):

    NAME = LXML
    ALTERNATE_NAMES = ["lxml-html"]

    features = ALTERNATE_NAMES + [NAME, HTML, FAST, PERMISSIVE]
    is_xml = False
    processing_instruction_class = ProcessingInstruction

    def default_parser(self, encoding):
        return etree.HTMLParser

    def feed(self, markup):
        encoding = self.soup.original_encoding
        try:
            self.parser = self.parser_for(encoding)
            self.parser.feed(markup)
            self.parser.close()
        except (UnicodeDecodeError, LookupError, etree.ParserError) as e:
            raise ParserRejectedMarkup(e)


    def test_fragment_to_document(self, fragment):
        """See `TreeBuilder`."""
        return '<html><body>%s</body></html>' % fragment
