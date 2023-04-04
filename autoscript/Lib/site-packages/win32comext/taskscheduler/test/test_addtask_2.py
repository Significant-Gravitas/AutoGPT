import time

import pythoncom
import win32api
from win32com.taskscheduler import taskscheduler

task_name = "test_addtask_2.job"
ts = pythoncom.CoCreateInstance(
    taskscheduler.CLSID_CTaskScheduler,
    None,
    pythoncom.CLSCTX_INPROC_SERVER,
    taskscheduler.IID_ITaskScheduler,
)
tasks = ts.Enum()
for task in tasks:
    print(task)
if task_name in tasks:
    print("Deleting existing task " + task_name)
    ts.Delete(task_name)

t = ts.NewWorkItem(task_name)
t.SetComment("Test a task running as local system acct")
t.SetApplicationName("c:\\python23\\python.exe")
t.SetPriority(taskscheduler.REALTIME_PRIORITY_CLASS)
t.SetParameters("test_localsystem.py")
t.SetWorkingDirectory("c:\\python23")
t.SetCreator("test_addtask_2.py")
t.SetMaxRunTime(20000)  # milliseconds
t.SetFlags(taskscheduler.TASK_FLAG_DELETE_WHEN_DONE)
t.SetAccountInformation(
    "", None
)  ## empty string for account name means to use local system
## None is only valid for local system acct or if task flags contain TASK_FLAG_RUN_ONLY_IF_LOGGED_ON

run_time = time.localtime(time.time() + 60)
tr_ind, tr = t.CreateTrigger()

tt = tr.GetTrigger()
tt.Flags = 0  ## flags for a new trigger default to TASK_TRIGGER_FLAG_DISABLED (4), make sure to clear them if not using any
tt.TriggerType = taskscheduler.TASK_TIME_TRIGGER_ONCE
tt.BeginYear = int(time.strftime("%Y", run_time))
tt.BeginMonth = int(time.strftime("%m", run_time))
tt.BeginDay = int(time.strftime("%d", run_time))
tt.StartMinute = int(time.strftime("%M", run_time))
tt.StartHour = int(time.strftime("%H", run_time))

tr.SetTrigger(tt)
print(t.GetTriggerString(tr_ind))

pf = t.QueryInterface(pythoncom.IID_IPersistFile)
pf.Save(None, 1)
