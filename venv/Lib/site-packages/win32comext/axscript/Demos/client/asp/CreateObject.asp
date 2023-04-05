<HTML>

<SCRIPT Language="Python" RUNAT=Server>

# Just for the sake of the demo, our Python script engine
# will create a Python.Interpreter COM object, and call that.

# This is completely useless, as the Python Script Engine is
# completely normal Python, and ASP does not impose retrictions, so
# there is nothing the COM object can do that we can not do natively.

o = Server.CreateObject("Python.Interpreter")

Response.Write("Python says 1+1=" + str(o.Eval("1+1")))

</SCRIPT>

</HTML>

