import os
import sys
import time

import pythoncom
import win32api
from win32com.taskscheduler import taskscheduler

task_name = "test_addtask.job"
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
t.SetComment("rude comments")
t.SetApplicationName(sys.executable)
t.SetPriority(taskscheduler.REALTIME_PRIORITY_CLASS)
t.SetParameters(
    "-c\"import win32ui,time;win32ui.MessageBox('hey bubba I am running');\""
)
t.SetWorkingDirectory(os.path.dirname(sys.executable))
t.SetCreator("test_addtask.py")
t.SetMaxRunTime(20000)  # milliseconds
t.SetFlags(
    taskscheduler.TASK_FLAG_INTERACTIVE | taskscheduler.TASK_FLAG_RUN_ONLY_IF_LOGGED_ON
)
##               |taskscheduler.TASK_FLAG_DELETE_WHEN_DONE)  #task self destructs when no more future run times
t.SetAccountInformation(win32api.GetUserName(), None)
## None is only valid for local system acct or if task flags contain TASK_FLAG_RUN_ONLY_IF_LOGGED_ON
t.SetWorkItemData("some binary garbage")

run_time = time.localtime(time.time() + 60)
tr_ind, tr = t.CreateTrigger()
tt = tr.GetTrigger()

## flags default to TASK_TRIGGER_FLAG_DISABLED (4)
tt.Flags = taskscheduler.TASK_TRIGGER_FLAG_KILL_AT_DURATION_END
tt.BeginYear = int(time.strftime("%Y", run_time))
tt.BeginMonth = int(time.strftime("%m", run_time))
tt.BeginDay = int(time.strftime("%d", run_time))
tt.StartMinute = int(time.strftime("%M", run_time))
tt.StartHour = int(time.strftime("%H", run_time))
tt.MinutesInterval = 1
tt.MinutesDuration = 5

tt.TriggerType = taskscheduler.TASK_TIME_TRIGGER_MONTHLYDATE
# months can contain multiples in a bitmask, use 1<<(month_nbr-1)
tt.MonthlyDate_Months = 1 << (
    int(time.strftime("%m", run_time)) - 1
)  ## corresponds to TASK_JANUARY..TASK_DECEMBER constants
# days too
tt.MonthlyDate_Days = 1 << (int(time.strftime("%d", run_time)) - 1)
tr.SetTrigger(tt)
print(t.GetTriggerString(tr_ind))

pf = t.QueryInterface(pythoncom.IID_IPersistFile)
pf.Save(None, 1)
