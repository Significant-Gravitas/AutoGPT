//Args:  input-file style-file output-file
var xml  = WScript.CreateObject("Microsoft.XMLDOM");          //input
xml.validateOnParse=false;
xml.load(WScript.Arguments(0));
var xsl  = WScript.CreateObject("Microsoft.XMLDOM");          //style
xsl.validateOnParse=false;
xsl.load(WScript.Arguments(1));
var out = WScript.CreateObject("Scripting.FileSystemObject"); //output
var replace = true; var unicode = false; //output file properties
var hdl = out.CreateTextFile( WScript.Arguments(2), replace, unicode )
hdl.write( xml.transformNode( xsl.documentElement ));
//eof
