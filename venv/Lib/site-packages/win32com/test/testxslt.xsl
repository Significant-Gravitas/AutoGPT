<!-- include in the xsl:stylesheet element:
     (a) the version attribute as usual
     (b) the XSLT namespace declaration as usual
     (c) the MSXSL namespace declaration
     (d) a namespace declaration to identify your functions
     (e) the 'extension-element-prefixes' attribute to give the
         namespace prefixes that indicate extension elements
         (i.e. 'msxsl')
     (f) the 'exclude-result-prefixes' attribute to indicate the
         namespaces that aren't supposed to be part of the result
         tree (i.e. 'foo') -->
<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:msxsl="urn:schemas-microsoft-com:xslt"
                xmlns:foo="http://www.pythoncom-test.com/foo"
                xmlns:bar="http://www.pythoncom-test.com/bar"
                extension-element-prefixes="msxsl"
                exclude-result-prefixes="foo bar">

<!-- do whatever output you want - you can use full XSLT functionality
     -->
<xsl:output method="html" />

<!-- define the Javascript functions that you want to include within
     a msxsl:script element.
     - language indicates the scripting language
     - implements-prefix gives the namespace prefix that you declared
       for your function (i.e. foo) -->
<msxsl:script language="javascript"
              implements-prefix="foo">
   function worked() {
      return "The jscript test worked";
    }
</msxsl:script>

<!-- ditto for Python, using the 'bar' namespace
-->
<msxsl:script language="python"
              implements-prefix="bar">
def worked():
  return "The Python test worked"
</msxsl:script>

<xsl:template match="/">
<!-- The output template.  Keep whitespace down as our test matches text exactly -->
<!-- call your functions using the prefix that you've used (i.e.
     foo) anywhere you can normally use an XPath function, but
     make sure it's returning the right kind of object -->
<xsl:value-of select="foo:worked()" />.
<xsl:value-of select="bar:worked()" />

</xsl:template>

</xsl:stylesheet>

