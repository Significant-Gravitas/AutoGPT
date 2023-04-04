# 'Request' example added jjk  11/20/98

import dde
import win32ui
from pywin.mfc import object


class MySystemTopic(object.Object):
    def __init__(self):
        object.Object.__init__(self, dde.CreateServerSystemTopic())

    def Exec(self, cmd):
        print("System Topic asked to exec", cmd)


class MyOtherTopic(object.Object):
    def __init__(self, topicName):
        object.Object.__init__(self, dde.CreateTopic(topicName))

    def Exec(self, cmd):
        print("Other Topic asked to exec", cmd)


class MyRequestTopic(object.Object):
    def __init__(self, topicName):
        topic = dde.CreateTopic(topicName)
        topic.AddItem(dde.CreateStringItem(""))
        object.Object.__init__(self, topic)

    def Request(self, aString):
        print("Request Topic asked to compute length of:", aString)
        return str(len(aString))


server = dde.CreateServer()
server.AddTopic(MySystemTopic())
server.AddTopic(MyOtherTopic("RunAnyCommand"))
server.AddTopic(MyRequestTopic("ComputeStringLength"))
server.Create("RunAny")

while 1:
    win32ui.PumpWaitingMessages(0, -1)
