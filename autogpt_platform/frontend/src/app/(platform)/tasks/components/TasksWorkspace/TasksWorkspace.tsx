"use client";

import { InsetSidePanel } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import type { ReactNode } from "react";
import type { TasksView } from "../../helpers";
import { useTasksPage } from "../../useTasksPage";
import { InboxPane } from "../InboxPane/InboxPane";
import { NewTaskDialog } from "../NewTaskDialog/NewTaskDialog";
import { TaskPanel } from "../TaskPanel/TaskPanel";
import { SheetCloseBar } from "./components/SheetCloseBar";

const PANEL_WIDTH = 420;

interface Props {
  className?: string;
  defaultView?: TasksView;
  /** Off where `?new=` already means something else (the home chat). */
  openCreateFromURL?: boolean;
  viewSwitch?: ReactNode;
}

export function TasksWorkspace({
  className,
  defaultView,
  openCreateFromURL,
  viewSwitch,
}: Props) {
  const page = useTasksPage({ defaultView, openCreateFromURL });

  return (
    <div className={className}>
      <InboxPane page={page} viewSwitch={viewSwitch} />
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
              taskId={page.taskID}
              onOpenTask={page.openTask}
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
