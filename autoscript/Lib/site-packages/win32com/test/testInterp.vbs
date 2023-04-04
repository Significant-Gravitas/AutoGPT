set o = CreateObject("Python.Interpreter")
if o.Eval("1+1") <> 2 Then
	WScript.Echo "Eval('1+1') failed"
	bFailed = True
end if

if bFailed then
	WScript.Echo "*********** VBScript tests failed *********"
else
	WScript.Echo "VBScript test worked OK"
end if

