from bs4.dammit import EntitySubstitution

class Formatter(EntitySubstitution):
    """Describes a strategy to use when outputting a parse tree to a string.

    Some parts of this strategy come from the distinction between
    HTML4, HTML5, and XML. Others are configurable by the user.

    Formatters are passed in as the `formatter` argument to methods
    like `PageElement.encode`. Most people won't need to think about
    formatters, and most people who need to think about them can pass
    in one of these predefined strings as `formatter` rather than
    making a new Formatter object:

    For HTML documents:
     * 'html' - HTML entity substitution for generic HTML documents. (default)
     * 'html5' - HTML entity substitution for HTML5 documents, as
                 well as some optimizations in the way tags are rendered.
     * 'minimal' - Only make the substitutions necessary to guarantee
                   valid HTML.
     * None - Do not perform any substitution. This will be faster
              but may result in invalid markup.

    For XML documents:
     * 'html' - Entity substitution for XHTML documents.
     * 'minimal' - Only make the substitutions necessary to guarantee
                   valid XML. (default)
     * None - Do not perform any substitution. This will be faster
              but may result in invalid markup.
    """
    # Registries of XML and HTML formatters.
    XML_FORMATTERS = {}
    HTML_FORMATTERS = {}

    HTML = 'html'
    XML = 'xml'

    HTML_DEFAULTS = dict(
        cdata_containing_tags=set(["script", "style"]),
    )

    def _default(self, language, value, kwarg):
        if value is not None:
            return value
        if language == self.XML:
            return set()
        return self.HTML_DEFAULTS[kwarg]

    def __init__(
            self, language=None, entity_substitution=None,
            void_element_close_prefix='/', cdata_containing_tags=None,
            empty_attributes_are_booleans=False, indent=1,
    ):
        """Constructor.

        :param language: This should be Formatter.XML if you are formatting
           XML markup and Formatter.HTML if you are formatting HTML markup.

        :param entity_substitution: A function to call to replace special
           characters with XML/HTML entities. For examples, see 
           bs4.dammit.EntitySubstitution.substitute_html and substitute_xml.
        :param void_element_close_prefix: By default, void elements
           are represented as <tag/> (XML rules) rather than <tag>
           (HTML rules). To get <tag>, pass in the empty string.
        :param cdata_containing_tags: The list of tags that are defined
           as containing CDATA in this dialect. For example, in HTML,
           <script> and <style> tags are defined as containing CDATA,
           and their contents should not be formatted.
        :param blank_attributes_are_booleans: Render attributes whose value
            is the empty string as HTML-style boolean attributes.
            (Attributes whose value is None are always rendered this way.)

        :param indent: If indent is a non-negative integer or string,
            then the contents of elements will be indented
            appropriately when pretty-printing. An indent level of 0,
            negative, or "" will only insert newlines. Using a
            positive integer indent indents that many spaces per
            level. If indent is a string (such as "\t"), that string
            is used to indent each level. The default behavior to
            indent one space per level.
        """
        self.language = language
        self.entity_substitution = entity_substitution
        self.void_element_close_prefix = void_element_close_prefix
        self.cdata_containing_tags = self._default(
            language, cdata_containing_tags, 'cdata_containing_tags'
        )
        self.empty_attributes_are_booleans=empty_attributes_are_booleans
        if indent is None:
            indent = 0
        if isinstance(indent, int):
            if indent < 0:
                indent = 0
            indent = ' ' * indent
        elif isinstance(indent, str):
            indent = indent
        else:
            indent = ' '
        self.indent = indent

    def substitute(self, ns):
        """Process a string that needs to undergo entity substitution.
        This may be a string encountered in an attribute value or as
        text.

        :param ns: A string.
        :return: A string with certain characters replaced by named
           or numeric entities.
        """
        if not self.entity_substitution:
            return ns
        from .element import NavigableString
        if (isinstance(ns, NavigableString)
            and ns.parent is not None
            and ns.parent.name in self.cdata_containing_tags):
            # Do nothing.
            return ns
        # Substitute.
        return self.entity_substitution(ns)

    def attribute_value(self, value):
        """Process the value of an attribute.

        :param ns: A string.
        :return: A string with certain characters replaced by named
           or numeric entities.
        """
        return self.substitute(value)
    
    def attributes(self, tag):
        """Reorder a tag's attributes however you want.
        
        By default, attributes are sorted alphabetically. This makes
        behavior consistent between Python 2 and Python 3, and preserves
        backwards compatibility with older versions of Beautiful Soup.

        If `empty_boolean_attributes` is True, then attributes whose
        values are set to the empty string will be treated as boolean
        attributes.
        """
        if tag.attrs is None:
            return []
        return sorted(
            (k, (None if self.empty_attributes_are_booleans and v == '' else v))
            for k, v in list(tag.attrs.items())
        )
   
class HTMLFormatter(Formatter):
    """A generic Formatter for HTML."""
    REGISTRY = {}
    def __init__(self, *args, **kwargs):
        super(HTMLFormatter, self).__init__(self.HTML, *args, **kwargs)

    
class XMLFormatter(Formatter):
    """A generic Formatter for XML."""
    REGISTRY = {}
    def __init__(self, *args, **kwargs):
        super(XMLFormatter, self).__init__(self.XML, *args, **kwargs)


# Set up aliases for the default formatters.
HTMLFormatter.REGISTRY['html'] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html
)
HTMLFormatter.REGISTRY["html5"] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html,
    void_element_close_prefix=None,
    empty_attributes_are_booleans=True,
)
HTMLFormatter.REGISTRY["minimal"] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_xml
)
HTMLFormatter.REGISTRY[None] = HTMLFormatter(
    entity_substitution=None
)
XMLFormatter.REGISTRY["html"] =  XMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html
)
XMLFormatter.REGISTRY["minimal"] = XMLFormatter(
    entity_substitution=EntitySubstitution.substitute_xml
)
XMLFormatter.REGISTRY[None] = Formatter(
    Formatter(Formatter.XML, entity_substitution=None)
)
