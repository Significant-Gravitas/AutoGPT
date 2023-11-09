' Test Pyhon.Dictionary using VBScript - this uses
' IDispatchEx, so is an interesting test.

set ob = CreateObject("Python.Dictionary")
ob("hello") = "there"
' Our keys are case insensitive.
ob.Item("hi") = ob("HELLO")

dim ok
ok = true

if ob("hello") <> "there" then
    WScript.Echo "**** The dictionary value was wrong!!"
    ok = false
end if

if ob("hi") <> "there" then
        WScript.Echo "**** The other dictionary value was wrong!!"
        ok = false
end if

if ok then
    WScript.Echo "VBScript has successfully tested Python.Dictionary"
end if


