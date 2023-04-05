import time

import pythoncom
import win32api
from win32com.taskscheduler import taskscheduler

test_task_name = "test_addtask_1.job"

ts = pythoncom.CoCreateInstance(
    taskscheduler.CLSID_CTaskScheduler,
    None,
    pythoncom.CLSCTX_INPROC_SERVER,
    taskscheduler.IID_ITaskScheduler,
)

tasks = ts.Enum()
for task in tasks:
    print(task)
if test_task_name in tasks:
    print("Deleting existing task " + test_task_name)
    ts.Delete(test_task_name)

new_task = pythoncom.CoCreateInstance(
    taskscheduler.CLSID_CTask,
    None,
    pythoncom.CLSCTX_INPROC_SERVER,
    taskscheduler.IID_ITask,
)
ts.AddWorkItem(test_task_name, new_task)  ## task object is modified in place

new_task.SetFlags(
    taskscheduler.TASK_FLAG_INTERACTIVE | taskscheduler.TASK_FLAG_RUN_ONLY_IF_LOGGED_ON
)
new_task.SetIdleWait(1, 10000)
new_task.SetComment("test task with idle trigger")
new_task.SetApplicationName("c:\\python23\\python.exe")
new_task.SetPriority(taskscheduler.REALTIME_PRIORITY_CLASS)
new_task.SetParameters(
    "-c\"import win32ui,time;win32ui.MessageBox('why aint you doing no work ?');\""
)
new_task.SetWorkingDirectory("c:\\python23")
new_task.SetCreator("test_addtask_1.py")
new_task.SetAccountInformation(win32api.GetUserName(), None)
## None is only valid for local system acct or if Flags contain TASK_FLAG_RUN_ONLY_IF_LOGGED_ON


run_time = time.localtime(time.time() + 30)
end_time = time.localtime(time.time() + 60 * 60 * 24)

tr_ind, tr = new_task.CreateTrigger()
tt = tr.GetTrigger()
tt.TriggerType = taskscheduler.TASK_EVENT_TRIGGER_ON_IDLE
tt.Flags = taskscheduler.TASK_TRIGGER_FLAG_HAS_END_DATE

tt.BeginYear = int(time.strftime("%Y", run_time))
tt.BeginMonth = int(time.strftime("%m", run_time))
tt.BeginDay = int(time.strftime("%d", run_time))
tt.StartMinute = int(time.strftime("%M", run_time))
tt.StartHour = int(time.strftime("%H", run_time))

tt.EndYear = int(time.strftime("%Y", end_time))
tt.EndMonth = int(time.strftime("%m", end_time))
tt.EndDay = int(time.strftime("%d", end_time))

tr.SetTrigger(tt)
print(new_task.GetTriggerString(tr_ind))

pf = new_task.QueryInterface(pythoncom.IID_IPersistFile)
pf.Save(None, 1)
