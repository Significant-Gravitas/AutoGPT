# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    'HTML5TreeBuilder',
    ]

import warnings
import re
from bs4.builder import (
    DetectsXMLParsedAsHTML,
    PERMISSIVE,
    HTML,
    HTML_5,
    HTMLTreeBuilder,
    )
from bs4.element import (
    NamespacedAttribute,
    nonwhitespace_re,
)
import html5lib
from html5lib.constants import (
    namespaces,
    prefixes,
    )
from bs4.element import (
    Comment,
    Doctype,
    NavigableString,
    Tag,
    )

try:
    # Pre-0.99999999
    from html5lib.treebuilders import _base as treebuilder_base
    new_html5lib = False
except ImportError as e:
    # 0.99999999 and up
    from html5lib.treebuilders import base as treebuilder_base
    new_html5lib = True

class HTML5TreeBuilder(HTMLTreeBuilder):
    """Use html5lib to build a tree.

    Note that this TreeBuilder does not support some features common
    to HTML TreeBuilders. Some of these features could theoretically
    be implemented, but at the very least it's quite difficult,
    because html5lib moves the parse tree around as it's being built.

    * This TreeBuilder doesn't use different subclasses of NavigableString
      based on the name of the tag in which the string was found.

    * You can't use a SoupStrainer to parse only part of a document.
    """

    NAME = "html5lib"

    features = [NAME, PERMISSIVE, HTML_5, HTML]

    # html5lib can tell us which line number and position in the
    # original file is the source of an element.
    TRACKS_LINE_NUMBERS = True
    
    def prepare_markup(self, markup, user_specified_encoding,
                       document_declared_encoding=None, exclude_encodings=None):
        # Store the user-specified encoding for use later on.
        self.user_specified_encoding = user_specified_encoding

        # document_declared_encoding and exclude_encodings aren't used
        # ATM because the html5lib TreeBuilder doesn't use
        # UnicodeDammit.
        if exclude_encodings:
            warnings.warn(
                "You provided a value for exclude_encoding, but the html5lib tree builder doesn't support exclude_encoding.",
                stacklevel=3
            )

        # html5lib only parses HTML, so if it's given XML that's worth
        # noting.
        DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup)

        yield (markup, None, None, False)

    # These methods are defined by Beautiful Soup.
    def feed(self, markup):
        if self.soup.parse_only is not None:
            warnings.warn(
                "You provided a value for parse_only, but the html5lib tree builder doesn't support parse_only. The entire document will be parsed.",
                stacklevel=4
            )
        parser = html5lib.HTMLParser(tree=self.create_treebuilder)
        self.underlying_builder.parser = parser
        extra_kwargs = dict()
        if not isinstance(markup, str):
            if new_html5lib:
                extra_kwargs['override_encoding'] = self.user_specified_encoding
            else:
                extra_kwargs['encoding'] = self.user_specified_encoding
        doc = parser.parse(markup, **extra_kwargs)
        
        # Set the character encoding detected by the tokenizer.
        if isinstance(markup, str):
            # We need to special-case this because html5lib sets
            # charEncoding to UTF-8 if it gets Unicode input.
            doc.original_encoding = None
        else:
            original_encoding = parser.tokenizer.stream.charEncoding[0]
            if not isinstance(original_encoding, str):
                # In 0.99999999 and up, the encoding is an html5lib
                # Encoding object. We want to use a string for compatibility
                # with other tree builders.
                original_encoding = original_encoding.name
            doc.original_encoding = original_encoding
        self.underlying_builder.parser = None
            
    def create_treebuilder(self, namespaceHTMLElements):
        self.underlying_builder = TreeBuilderForHtml5lib(
            namespaceHTMLElements, self.soup,
            store_line_numbers=self.store_line_numbers
        )
        return self.underlying_builder

    def test_fragment_to_document(self, fragment):
        """See `TreeBuilder`."""
        return '<html><head></head><body>%s</body></html>' % fragment


class TreeBuilderForHtml5lib(treebuilder_base.TreeBuilder):
    
    def __init__(self, namespaceHTMLElements, soup=None,
                 store_line_numbers=True, **kwargs):
        if soup:
            self.soup = soup
        else:
            from bs4 import BeautifulSoup
            # TODO: Why is the parser 'html.parser' here? To avoid an
            # infinite loop?
            self.soup = BeautifulSoup(
                "", "html.parser", store_line_numbers=store_line_numbers,
                **kwargs
            )
        # TODO: What are **kwargs exactly? Should they be passed in
        # here in addition to/instead of being passed to the BeautifulSoup
        # constructor?
        super(TreeBuilderForHtml5lib, self).__init__(namespaceHTMLElements)

        # This will be set later to an html5lib.html5parser.HTMLParser
        # object, which we can use to track the current line number.
        self.parser = None
        self.store_line_numbers = store_line_numbers
        
    def documentClass(self):
        self.soup.reset()
        return Element(self.soup, self.soup, None)

    def insertDoctype(self, token):
        name = token["name"]
        publicId = token["publicId"]
        systemId = token["systemId"]

        doctype = Doctype.for_name_and_ids(name, publicId, systemId)
        self.soup.object_was_parsed(doctype)

    def elementClass(self, name, namespace):
        kwargs = {}
        if self.parser and self.store_line_numbers:
            # This represents the point immediately after the end of the
            # tag. We don't know when the tag started, but we do know
            # where it ended -- the character just before this one.
            sourceline, sourcepos = self.parser.tokenizer.stream.position()
            kwargs['sourceline'] = sourceline
            kwargs['sourcepos'] = sourcepos-1
        tag = self.soup.new_tag(name, namespace, **kwargs)

        return Element(tag, self.soup, namespace)

    def commentClass(self, data):
        return TextNode(Comment(data), self.soup)

    def fragmentClass(self):
        from bs4 import BeautifulSoup
        # TODO: Why is the parser 'html.parser' here? To avoid an
        # infinite loop?
        self.soup = BeautifulSoup("", "html.parser")
        self.soup.name = "[document_fragment]"
        return Element(self.soup, self.soup, None)

    def appendChild(self, node):
        # XXX This code is not covered by the BS4 tests.
        self.soup.append(node.element)

    def getDocument(self):
        return self.soup

    def getFragment(self):
        return treebuilder_base.TreeBuilder.getFragment(self).element

    def testSerializer(self, element):
        from bs4 import BeautifulSoup
        rv = []
        doctype_re = re.compile(r'^(.*?)(?: PUBLIC "(.*?)"(?: "(.*?)")?| SYSTEM "(.*?)")?$')

        def serializeElement(element, indent=0):
            if isinstance(element, BeautifulSoup):
                pass
            if isinstance(element, Doctype):
                m = doctype_re.match(element)
                if m:
                    name = m.group(1)
                    if m.lastindex > 1:
                        publicId = m.group(2) or ""
                        systemId = m.group(3) or m.group(4) or ""
                        rv.append("""|%s<!DOCTYPE %s "%s" "%s">""" %
                                  (' ' * indent, name, publicId, systemId))
                    else:
                        rv.append("|%s<!DOCTYPE %s>" % (' ' * indent, name))
                else:
                    rv.append("|%s<!DOCTYPE >" % (' ' * indent,))
            elif isinstance(element, Comment):
                rv.append("|%s<!-- %s -->" % (' ' * indent, element))
            elif isinstance(element, NavigableString):
                rv.append("|%s\"%s\"" % (' ' * indent, element))
            else:
                if element.namespace:
                    name = "%s %s" % (prefixes[element.namespace],
                                      element.name)
                else:
                    name = element.name
                rv.append("|%s<%s>" % (' ' * indent, name))
                if element.attrs:
                    attributes = []
                    for name, value in list(element.attrs.items()):
                        if isinstance(name, NamespacedAttribute):
                            name = "%s %s" % (prefixes[name.namespace], name.name)
                        if isinstance(value, list):
                            value = " ".join(value)
                        attributes.append((name, value))

                    for name, value in sorted(attributes):
                        rv.append('|%s%s="%s"' % (' ' * (indent + 2), name, value))
                indent += 2
                for child in element.children:
                    serializeElement(child, indent)
        serializeElement(element, 0)

        return "\n".join(rv)

class AttrList(object):
    def __init__(self, element):
        self.element = element
        self.attrs = dict(self.element.attrs)
    def __iter__(self):
        return list(self.attrs.items()).__iter__()
    def __setitem__(self, name, value):
        # If this attribute is a multi-valued attribute for this element,
        # turn its value into a list.
        list_attr = self.element.cdata_list_attributes or {}
        if (name in list_attr.get('*', [])
            or (self.element.name in list_attr
                and name in list_attr.get(self.element.name, []))):
            # A node that is being cloned may have already undergone
            # this procedure.
            if not isinstance(value, list):
                value = nonwhitespace_re.findall(value)
        self.element[name] = value
    def items(self):
        return list(self.attrs.items())
    def keys(self):
        return list(self.attrs.keys())
    def __len__(self):
        return len(self.attrs)
    def __getitem__(self, name):
        return self.attrs[name]
    def __contains__(self, name):
        return name in list(self.attrs.keys())


class Element(treebuilder_base.Node):
    def __init__(self, element, soup, namespace):
        treebuilder_base.Node.__init__(self, element.name)
        self.element = element
        self.soup = soup
        self.namespace = namespace

    def appendChild(self, node):
        string_child = child = None
        if isinstance(node, str):
            # Some other piece of code decided to pass in a string
            # instead of creating a TextElement object to contain the
            # string.
            string_child = child = node
        elif isinstance(node, Tag):
            # Some other piece of code decided to pass in a Tag
            # instead of creating an Element object to contain the
            # Tag.
            child = node
        elif node.element.__class__ == NavigableString:
            string_child = child = node.element
            node.parent = self
        else:
            child = node.element
            node.parent = self

        if not isinstance(child, str) and child.parent is not None:
            node.element.extract()

        if (string_child is not None and self.element.contents
            and self.element.contents[-1].__class__ == NavigableString):
            # We are appending a string onto another string.
            # TODO This has O(n^2) performance, for input like
            # "a</a>a</a>a</a>..."
            old_element = self.element.contents[-1]
            new_element = self.soup.new_string(old_element + string_child)
            old_element.replace_with(new_element)
            self.soup._most_recent_element = new_element
        else:
            if isinstance(node, str):
                # Create a brand new NavigableString from this string.
                child = self.soup.new_string(node)

            # Tell Beautiful Soup to act as if it parsed this element
            # immediately after the parent's last descendant. (Or
            # immediately after the parent, if it has no children.)
            if self.element.contents:
                most_recent_element = self.element._last_descendant(False)
            elif self.element.next_element is not None:
                # Something from further ahead in the parse tree is
                # being inserted into this earlier element. This is
                # very annoying because it means an expensive search
                # for the last element in the tree.
                most_recent_element = self.soup._last_descendant()
            else:
                most_recent_element = self.element

            self.soup.object_was_parsed(
                child, parent=self.element,
                most_recent_element=most_recent_element)

    def getAttributes(self):
        if isinstance(self.element, Comment):
            return {}
        return AttrList(self.element)

    def setAttributes(self, attributes):
        if attributes is not None and len(attributes) > 0:
            converted_attributes = []
            for name, value in list(attributes.items()):
                if isinstance(name, tuple):
                    new_name = NamespacedAttribute(*name)
                    del attributes[name]
                    attributes[new_name] = value

            self.soup.builder._replace_cdata_list_attribute_values(
                self.name, attributes)
            for name, value in list(attributes.items()):
                self.element[name] = value

            # The attributes may contain variables that need substitution.
            # Call set_up_substitutions manually.
            #
            # The Tag constructor called this method when the Tag was created,
            # but we just set/changed the attributes, so call it again.
            self.soup.builder.set_up_substitutions(self.element)
    attributes = property(getAttributes, setAttributes)

    def insertText(self, data, insertBefore=None):
        text = TextNode(self.soup.new_string(data), self.soup)
        if insertBefore:
            self.insertBefore(text, insertBefore)
        else:
            self.appendChild(text)

    def insertBefore(self, node, refNode):
        index = self.element.index(refNode.element)
        if (node.element.__class__ == NavigableString and self.element.contents
            and self.element.contents[index-1].__class__ == NavigableString):
            # (See comments in appendChild)
            old_node = self.element.contents[index-1]
            new_str = self.soup.new_string(old_node + node.element)
            old_node.replace_with(new_str)
        else:
            self.element.insert(index, node.element)
            node.parent = self

    def removeChild(self, node):
        node.element.extract()

    def reparentChildren(self, new_parent):
        """Move all of this tag's children into another tag."""
        # print("MOVE", self.element.contents)
        # print("FROM", self.element)
        # print("TO", new_parent.element)

        element = self.element
        new_parent_element = new_parent.element
        # Determine what this tag's next_element will be once all the children
        # are removed.
        final_next_element = element.next_sibling

        new_parents_last_descendant = new_parent_element._last_descendant(False, False)
        if len(new_parent_element.contents) > 0:
            # The new parent already contains children. We will be
            # appending this tag's children to the end.
            new_parents_last_child = new_parent_element.contents[-1]
            new_parents_last_descendant_next_element = new_parents_last_descendant.next_element
        else:
            # The new parent contains no children.
            new_parents_last_child = None
            new_parents_last_descendant_next_element = new_parent_element.next_element

        to_append = element.contents
        if len(to_append) > 0:
            # Set the first child's previous_element and previous_sibling
            # to elements within the new parent
            first_child = to_append[0]
            if new_parents_last_descendant is not None:
                first_child.previous_element = new_parents_last_descendant
            else:
                first_child.previous_element = new_parent_element
            first_child.previous_sibling = new_parents_last_child
            if new_parents_last_descendant is not None:
                new_parents_last_descendant.next_element = first_child
            else:
                new_parent_element.next_element = first_child
            if new_parents_last_child is not None:
                new_parents_last_child.next_sibling = first_child

            # Find the very last element being moved. It is now the
            # parent's last descendant. It has no .next_sibling and
            # its .next_element is whatever the previous last
            # descendant had.
            last_childs_last_descendant = to_append[-1]._last_descendant(False, True)

            last_childs_last_descendant.next_element = new_parents_last_descendant_next_element
            if new_parents_last_descendant_next_element is not None:
                # TODO: This code has no test coverage and I'm not sure
                # how to get html5lib to go through this path, but it's
                # just the other side of the previous line.
                new_parents_last_descendant_next_element.previous_element = last_childs_last_descendant
            last_childs_last_descendant.next_sibling = None

        for child in to_append:
            child.parent = new_parent_element
            new_parent_element.contents.append(child)

        # Now that this element has no children, change its .next_element.
        element.contents = []
        element.next_element = final_next_element

        # print("DONE WITH MOVE")
        # print("FROM", self.element)
        # print("TO", new_parent_element)

    def cloneNode(self):
        tag = self.soup.new_tag(self.element.name, self.namespace)
        node = Element(tag, self.soup, self.namespace)
        for key,value in self.attributes:
            node.attributes[key] = value
        return node

    def hasContent(self):
        return self.element.contents

    def getNameTuple(self):
        if self.namespace == None:
            return namespaces["html"], self.name
        else:
            return self.namespace, self.name

    nameTuple = property(getNameTuple)

class TextNode(Element):
    def __init__(self, element, soup):
        treebuilder_base.Node.__init__(self, None)
        self.element = element
        self.soup = soup

    def cloneNode(self):
        raise NotImplementedError
