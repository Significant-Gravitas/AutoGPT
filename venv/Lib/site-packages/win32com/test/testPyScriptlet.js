function print(msg)
{
  WScript.Echo(msg) ;
}

function check(condition, msg)
{
  if (!condition) {
    print("***** testPyScriptlet.js failed *****");
    print(msg);
  }
}

var thisScriptEngine = ScriptEngine() ;

var majorVersion = ScriptEngineMajorVersion() ;
var minorVersion = ScriptEngineMinorVersion() ;
var buildVersion = ScriptEngineBuildVersion() ;

WScript.Echo(thisScriptEngine + " Version " + majorVersion + "." + minorVersion + " Build " + buildVersion) ;

var scriptlet = new  ActiveXObject("TestPys.Scriptlet") ;

check(scriptlet.PyProp1=="PyScript Property1", "PyProp1 wasn't correct initial value");
scriptlet.PyProp1 = "New Value";
check(scriptlet.PyProp1=="New Value", "PyProp1 wasn't correct new value");

check(scriptlet.PyProp2=="PyScript Property2", "PyProp2 wasn't correct initial value");
scriptlet.PyProp2 = "Another New Value";
check(scriptlet.PyProp2=="Another New Value", "PyProp2 wasn't correct new value");

check(scriptlet.PyMethod1()=="PyMethod1 called", "Method1 wrong value");
check(scriptlet.PyMethod2()=="PyMethod2 called", "Method2 wrong value");
