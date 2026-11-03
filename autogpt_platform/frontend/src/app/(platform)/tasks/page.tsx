"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { BoardView } from "./components/BoardView/BoardView";
import { FilterChips } from "./components/FilterChips/FilterChips";
import { ListView } from "./components/ListView/ListView";
import { NewTaskDialog } from "./components/NewTaskDialog/NewTaskDialog";
import { TasksHeader } from "./components/TasksHeader/TasksHeader";
import { TaskSheet } from "./components/TaskSheet/TaskSheet";
import { useTasksPage } from "./useTasksPage";

export default function TasksPage() {
  const page = useTasksPage();

  return (
    <div className="flex min-h-full w-full flex-col gap-4 px-4 py-6 md:px-8">
      <TasksHeader
        view={page.view}
        onViewChange={page.setView}
        filters={page.filters}
        onFiltersChange={page.setFilters}
        experts={page.experts}
        showHidden={page.showHidden}
        onShowHiddenChange={page.setShowHidden}
        onNewTask={() => page.setIsCreateOpen(true)}
      />
      <FilterChips chips={page.filterChips} onChange={page.setFilters} />
      {page.error ? (
        <ErrorCard
          context="tasks"
          responseError={{ message: page.error.message }}
          onRetry={() => page.refetch()}
        />
      ) : page.view === "board" ? (
        <BoardView
          tasks={page.tasks}
          isLoading={page.isLoading}
          showHidden={page.showHidden}
          onOpenTask={page.openTask}
        />
      ) : (
        <ListView
          tasks={page.tasks}
          isLoading={page.isLoading}
          onOpenTask={page.openTask}
          emptyLabel={
            page.view === "mine"
              ? "Nothing assigned to you right now."
              : "No tasks match these filters."
          }
        />
      )}
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
