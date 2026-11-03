"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { describeLiveCounts } from "./helpers";
import { InboxBoard } from "./components/InboxBoard/InboxBoard";
import { InboxEmpty } from "./components/InboxEmpty/InboxEmpty";
import { InboxFilters } from "./components/InboxFilters/InboxFilters";
import { InboxHeader } from "./components/InboxHeader/InboxHeader";
import { InboxList } from "./components/InboxList/InboxList";
import { InboxListSkeleton } from "./components/InboxList/InboxListSkeleton";
import { InboxTray } from "./components/InboxList/InboxTray";
import { LiveRegion } from "./components/LiveRegion/LiveRegion";
import { NewTaskDialog } from "./components/NewTaskDialog/NewTaskDialog";
import { TaskSheet } from "./components/TaskSheet/TaskSheet";
import { useContainerWidth } from "./useContainerWidth";
import { useTasksPage } from "./useTasksPage";

export default function TasksPage() {
  const page = useTasksPage();
  const { attach, isWide } = useContainerWidth();

  return (
    <div
      ref={attach}
      className="mx-auto flex min-h-full w-full max-w-6xl flex-col gap-4 px-4 py-6 md:px-8"
    >
      <InboxHeader
        view={page.view}
        onViewChange={page.setView}
        onNewTask={() => page.setIsCreateOpen(true)}
      />
      <InboxFilters
        tab={page.tab}
        onTabChange={page.setTab}
        counts={page.counts}
        mine={page.mine}
        onMineChange={page.setMine}
        filters={page.filters}
        onFiltersChange={page.setFilters}
        chips={page.filterChips}
        experts={page.experts}
      />
      <LiveRegion message={describeLiveCounts(page.counts)} />
      <InboxBody page={page} isBoard={page.view === "board" && isWide} />
      <NewTaskDialog
        isOpen={page.isCreateOpen}
        onOpenChange={page.setIsCreateOpen}
        experts={page.experts}
        onCreated={page.openTask}
      />
      <TaskSheet
        taskId={page.taskID}
        onOpenChange={page.handleSheetOpenChange}
        onOpenTask={page.openTask}
      />
    </div>
  );
}

interface BodyProps {
  page: ReturnType<typeof useTasksPage>;
  isBoard: boolean;
}

function InboxBody({ page, isBoard }: BodyProps) {
  if (page.error) {
    return (
      <ErrorCard
        context="tasks"
        responseError={{ message: page.error.message }}
        onRetry={() => page.refetch()}
      />
    );
  }
  if (page.isLoading) return <InboxListSkeleton />;
  if (page.tasks.length === 0) {
    return (
      <InboxTray>
        <InboxEmpty
          tab={page.tab}
          hasFilters={page.filterChips.length > 0 || page.mine}
          onClearFilters={page.clearFilters}
        />
      </InboxTray>
    );
  }
  if (isBoard) {
    return (
      <InboxBoard
        tasks={page.tasks}
        tab={page.tab}
        expandedID={page.expandedID}
        onToggle={page.toggleRow}
        onOpen={page.openTask}
      />
    );
  }
  return (
    <InboxList
      tasks={page.tasks}
      expandedID={page.expandedID}
      onToggle={page.toggleRow}
      onOpen={page.openTask}
    />
  );
}
