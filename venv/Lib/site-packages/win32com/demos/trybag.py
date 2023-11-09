import pythoncom
from win32com.server import exception, util

VT_EMPTY = pythoncom.VT_EMPTY


class Bag:
    _public_methods_ = ["Read", "Write"]
    _com_interfaces_ = [pythoncom.IID_IPropertyBag]

    def __init__(self):
        self.data = {}

    def Read(self, propName, varType, errorLog):
        print("read: name=", propName, "type=", varType)
        if propName not in self.data:
            if errorLog:
                hr = 0x80070057
                exc = pythoncom.com_error(0, "Bag.Read", "no such item", None, 0, hr)
                errorLog.AddError(propName, exc)
            raise exception.Exception(scode=hr)
        return self.data[propName]

    def Write(self, propName, value):
        print("write: name=", propName, "value=", value)
        self.data[propName] = value


class Target:
    _public_methods_ = ["GetClassID", "InitNew", "Load", "Save"]
    _com_interfaces_ = [pythoncom.IID_IPersist, pythoncom.IID_IPersistPropertyBag]

    def GetClassID(self):
        raise exception.Exception(scode=0x80004005)  # E_FAIL

    def InitNew(self):
        pass

    def Load(self, bag, log):
        print(bag.Read("prop1", VT_EMPTY, log))
        print(bag.Read("prop2", VT_EMPTY, log))
        try:
            print(bag.Read("prop3", VT_EMPTY, log))
        except exception.Exception:
            pass

    def Save(self, bag, clearDirty, saveAllProps):
        bag.Write("prop1", "prop1.hello")
        bag.Write("prop2", "prop2.there")


class Log:
    _public_methods_ = ["AddError"]
    _com_interfaces_ = [pythoncom.IID_IErrorLog]

    def AddError(self, propName, excepInfo):
        print("error: propName=", propName, "error=", excepInfo)


def test():
    bag = Bag()
    target = Target()
    log = Log()

    target.Save(bag, 1, 1)
    target.Load(bag, log)

    comBag = util.wrap(bag, pythoncom.IID_IPropertyBag)
    comTarget = util.wrap(target, pythoncom.IID_IPersistPropertyBag)
    comLog = util.wrap(log, pythoncom.IID_IErrorLog)

    comTarget.Save(comBag, 1, 1)
    comTarget.Load(comBag, comLog)


if __name__ == "__main__":
    test()
