"use client";

import { InboxPane } from "./components/InboxPane/InboxPane";
import { NewTaskDialog } from "./components/NewTaskDialog/NewTaskDialog";
import { SplitView } from "./components/SplitView/SplitView";
import { TaskPanel } from "./components/TaskPanel/TaskPanel";
import { useTasksPage } from "./useTasksPage";

export default function TasksPage() {
  const page = useTasksPage();

  return (
    <div className="flex h-[calc(100svh-4.5rem)] min-h-0 w-full flex-col px-4 pb-4 md:h-svh md:px-6 md:pt-6">
      <SplitView
        selectedKey={page.taskID}
        onBack={page.closeTask}
        list={<InboxPane page={page} />}
        panel={
          page.taskID ? (
            <TaskPanel
              key={page.taskID}
              taskId={page.taskID}
              onClose={page.closeTask}
              onOpenTask={page.openTask}
            />
          ) : null
        }
      />
      <NewTaskDialog
        isOpen={page.isCreateOpen}
        onOpenChange={page.setIsCreateOpen}
        experts={page.experts}
        onCreated={page.openTask}
      />
    </div>
  );
}
