<%@ Language=Python %>
<HTML>

<HEAD>

<BODY BACKGROUND="/samples/images/backgrnd.gif">

<TITLE>Python test</TITLE>

</HEAD>

<BODY BGCOLOR="FFFFFF">

<SCRIPT Language="Python" RUNAT=Server>
# NOTE that the <% tags below execute _before_ these tags!
Response.Write("Hello from Python<P>")
Response.Write("Browser is "+bc.browser)
import win32api # Should be no problem using win32api in ASP pages.
Response.Write("<p>Win32 username is "+win32api.GetUserName())
</SCRIPT>

<BODY BGCOLOR="FFFFFF">

<% 
import sys
print sys.path
from win32com.axscript.asputil import *
print "Hello"
print "There"
print "How are you"
%>

<%bc = Server.CreateObject("MSWC.BrowserType")%>
<BODY BGCOLOR="FFFFFF">
<table border=1> 
<tr><td>Browser</td><td> <%=bc.browser %> 
<tr><td>Version</td><td> <%=bc.version %> </td></TR> 
<tr><td>Frames</td><td> 
<%Response.Write( iif(bc.frames, "TRUE", "FALSE")) %></td></TR> 
<tr><td>Tables</td><td> 
<%Response.Write( iif (bc.tables, "TRUE", "FALSE")) %></td></TR> 
<tr><td>BackgroundSounds</td><td> 
<%Response.Write( iif(bc.BackgroundSounds, "TRUE", "FALSE"))%></td></TR> 
<tr><td>VBScript</td><td> 
<%Response.Write( iif(bc.vbscript, "TRUE", "FALSE"))%></td></TR> 
<tr><td>JavaScript</td><td> 
<%Response.Write( iif(bc.javascript, "TRUE", "FALSE"))%></td></TR> 

</table> 

</body>
</html>
