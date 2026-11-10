"use client";

import { InsetSidePanel } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { NewTaskDialog } from "./components/NewTaskDialog/NewTaskDialog";
import { SheetCloseBar } from "./components/SheetCloseBar/SheetCloseBar";
import { StatusTabs } from "./components/StatusTabs/StatusTabs";
import { TaskPanel } from "./components/TaskPanel/TaskPanel";
import { TasksBody } from "./components/TasksBody/TasksBody";
import { TasksHeader } from "./components/TasksHeader/TasksHeader";
import { useTasksPage } from "./useTasksPage";

const PANEL_WIDTH = 420;

export default function TasksPage() {
  const page = useTasksPage();

  return (
    <div className="flex min-h-0 w-full flex-col px-4 pb-4 md:px-6 md:pt-6">
      <div className="mx-auto flex w-full max-w-6xl shrink-0 flex-col gap-4 pb-4">
        <TasksHeader
          view={page.view}
          onViewChange={page.setView}
          onNewTask={() => page.setIsCreateOpen(true)}
        />
        <StatusTabs
          tab={page.tab}
          onTabChange={page.setTab}
          tasks={page.allTasks}
        />
      </div>
      <div className="mx-auto w-full max-w-6xl">
        <TasksBody page={page} onNewTask={() => page.setIsCreateOpen(true)} />
      </div>
      <InsetSidePanel
        isOpen={!!page.taskID}
        title="Task"
        ariaLabel="Task details"
        closeLabel="Close task"
        onClose={page.closeTask}
        width={PANEL_WIDTH}
      >
        {page.taskID ? (
          <>
            <SheetCloseBar onClose={page.closeTask} />
            <TaskPanel
              key={page.taskID}
              taskID={page.taskID}
              experts={page.experts}
              onArchived={page.closeTask}
            />
          </>
        ) : null}
      </InsetSidePanel>
      <NewTaskDialog
        isOpen={page.isCreateOpen}
        onOpenChange={page.setIsCreateOpen}
        experts={page.experts}
        onCreated={page.openTask}
      />
    </div>
  );
}
