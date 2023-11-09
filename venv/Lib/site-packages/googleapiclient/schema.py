# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Schema processing for discovery based APIs

Schemas holds an APIs discovery schemas. It can return those schema as
deserialized JSON objects, or pretty print them as prototype objects that
conform to the schema.

For example, given the schema:

 schema = \"\"\"{
   "Foo": {
    "type": "object",
    "properties": {
     "etag": {
      "type": "string",
      "description": "ETag of the collection."
     },
     "kind": {
      "type": "string",
      "description": "Type of the collection ('calendar#acl').",
      "default": "calendar#acl"
     },
     "nextPageToken": {
      "type": "string",
      "description": "Token used to access the next
         page of this result. Omitted if no further results are available."
     }
    }
   }
 }\"\"\"

 s = Schemas(schema)
 print s.prettyPrintByName('Foo')

 Produces the following output:

  {
   "nextPageToken": "A String", # Token used to access the
       # next page of this result. Omitted if no further results are available.
   "kind": "A String", # Type of the collection ('calendar#acl').
   "etag": "A String", # ETag of the collection.
  },

The constructor takes a discovery document in which to look up named schema.
"""
from __future__ import absolute_import

# TODO(jcgregorio) support format, enum, minimum, maximum

__author__ = "jcgregorio@google.com (Joe Gregorio)"


from collections import OrderedDict

from googleapiclient import _helpers as util


class Schemas(object):
    """Schemas for an API."""

    def __init__(self, discovery):
        """Constructor.

        Args:
          discovery: object, Deserialized discovery document from which we pull
            out the named schema.
        """
        self.schemas = discovery.get("schemas", {})

        # Cache of pretty printed schemas.
        self.pretty = {}

    @util.positional(2)
    def _prettyPrintByName(self, name, seen=None, dent=0):
        """Get pretty printed object prototype from the schema name.

        Args:
          name: string, Name of schema in the discovery document.
          seen: list of string, Names of schema already seen. Used to handle
            recursive definitions.

        Returns:
          string, A string that contains a prototype object with
            comments that conforms to the given schema.
        """
        if seen is None:
            seen = []

        if name in seen:
            # Do not fall into an infinite loop over recursive definitions.
            return "# Object with schema name: %s" % name
        seen.append(name)

        if name not in self.pretty:
            self.pretty[name] = _SchemaToStruct(
                self.schemas[name], seen, dent=dent
            ).to_str(self._prettyPrintByName)

        seen.pop()

        return self.pretty[name]

    def prettyPrintByName(self, name):
        """Get pretty printed object prototype from the schema name.

        Args:
          name: string, Name of schema in the discovery document.

        Returns:
          string, A string that contains a prototype object with
            comments that conforms to the given schema.
        """
        # Return with trailing comma and newline removed.
        return self._prettyPrintByName(name, seen=[], dent=0)[:-2]

    @util.positional(2)
    def _prettyPrintSchema(self, schema, seen=None, dent=0):
        """Get pretty printed object prototype of schema.

        Args:
          schema: object, Parsed JSON schema.
          seen: list of string, Names of schema already seen. Used to handle
            recursive definitions.

        Returns:
          string, A string that contains a prototype object with
            comments that conforms to the given schema.
        """
        if seen is None:
            seen = []

        return _SchemaToStruct(schema, seen, dent=dent).to_str(self._prettyPrintByName)

    def prettyPrintSchema(self, schema):
        """Get pretty printed object prototype of schema.

        Args:
          schema: object, Parsed JSON schema.

        Returns:
          string, A string that contains a prototype object with
            comments that conforms to the given schema.
        """
        # Return with trailing comma and newline removed.
        return self._prettyPrintSchema(schema, dent=0)[:-2]

    def get(self, name, default=None):
        """Get deserialized JSON schema from the schema name.

        Args:
          name: string, Schema name.
          default: object, return value if name not found.
        """
        return self.schemas.get(name, default)


class _SchemaToStruct(object):
    """Convert schema to a prototype object."""

    @util.positional(3)
    def __init__(self, schema, seen, dent=0):
        """Constructor.

        Args:
          schema: object, Parsed JSON schema.
          seen: list, List of names of schema already seen while parsing. Used to
            handle recursive definitions.
          dent: int, Initial indentation depth.
        """
        # The result of this parsing kept as list of strings.
        self.value = []

        # The final value of the parsing.
        self.string = None

        # The parsed JSON schema.
        self.schema = schema

        # Indentation level.
        self.dent = dent

        # Method that when called returns a prototype object for the schema with
        # the given name.
        self.from_cache = None

        # List of names of schema already seen while parsing.
        self.seen = seen

    def emit(self, text):
        """Add text as a line to the output.

        Args:
          text: string, Text to output.
        """
        self.value.extend(["  " * self.dent, text, "\n"])

    def emitBegin(self, text):
        """Add text to the output, but with no line terminator.

        Args:
          text: string, Text to output.
        """
        self.value.extend(["  " * self.dent, text])

    def emitEnd(self, text, comment):
        """Add text and comment to the output with line terminator.

        Args:
          text: string, Text to output.
          comment: string, Python comment.
        """
        if comment:
            divider = "\n" + "  " * (self.dent + 2) + "# "
            lines = comment.splitlines()
            lines = [x.rstrip() for x in lines]
            comment = divider.join(lines)
            self.value.extend([text, " # ", comment, "\n"])
        else:
            self.value.extend([text, "\n"])

    def indent(self):
        """Increase indentation level."""
        self.dent += 1

    def undent(self):
        """Decrease indentation level."""
        self.dent -= 1

    def _to_str_impl(self, schema):
        """Prototype object based on the schema, in Python code with comments.

        Args:
          schema: object, Parsed JSON schema file.

        Returns:
          Prototype object based on the schema, in Python code with comments.
        """
        stype = schema.get("type")
        if stype == "object":
            self.emitEnd("{", schema.get("description", ""))
            self.indent()
            if "properties" in schema:
                properties = schema.get("properties", {})
                sorted_properties = OrderedDict(sorted(properties.items()))
                for pname, pschema in sorted_properties.items():
                    self.emitBegin('"%s": ' % pname)
                    self._to_str_impl(pschema)
            elif "additionalProperties" in schema:
                self.emitBegin('"a_key": ')
                self._to_str_impl(schema["additionalProperties"])
            self.undent()
            self.emit("},")
        elif "$ref" in schema:
            schemaName = schema["$ref"]
            description = schema.get("description", "")
            s = self.from_cache(schemaName, seen=self.seen)
            parts = s.splitlines()
            self.emitEnd(parts[0], description)
            for line in parts[1:]:
                self.emit(line.rstrip())
        elif stype == "boolean":
            value = schema.get("default", "True or False")
            self.emitEnd("%s," % str(value), schema.get("description", ""))
        elif stype == "string":
            value = schema.get("default", "A String")
            self.emitEnd('"%s",' % str(value), schema.get("description", ""))
        elif stype == "integer":
            value = schema.get("default", "42")
            self.emitEnd("%s," % str(value), schema.get("description", ""))
        elif stype == "number":
            value = schema.get("default", "3.14")
            self.emitEnd("%s," % str(value), schema.get("description", ""))
        elif stype == "null":
            self.emitEnd("None,", schema.get("description", ""))
        elif stype == "any":
            self.emitEnd('"",', schema.get("description", ""))
        elif stype == "array":
            self.emitEnd("[", schema.get("description"))
            self.indent()
            self.emitBegin("")
            self._to_str_impl(schema["items"])
            self.undent()
            self.emit("],")
        else:
            self.emit("Unknown type! %s" % stype)
            self.emitEnd("", "")

        self.string = "".join(self.value)
        return self.string

    def to_str(self, from_cache):
        """Prototype object based on the schema, in Python code with comments.

        Args:
          from_cache: callable(name, seen), Callable that retrieves an object
             prototype for a schema with the given name. Seen is a list of schema
             names already seen as we recursively descend the schema definition.

        Returns:
          Prototype object based on the schema, in Python code with comments.
          The lines of the code will all be properly indented.
        """
        self.from_cache = from_cache
        return self._to_str_impl(self.schema)
