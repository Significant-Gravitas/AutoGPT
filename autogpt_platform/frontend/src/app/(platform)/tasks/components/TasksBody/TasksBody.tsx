import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import type { useTasksPage } from "../../useTasksPage";
import { TaskBoard } from "../TaskBoard/TaskBoard";
import { TaskList } from "../TaskList/TaskList";
import { TaskListSkeleton } from "../TaskList/TaskListSkeleton";
import { TasksCard } from "../TaskList/TasksCard";
import { TasksEmpty } from "../TasksEmpty/TasksEmpty";

interface Props {
  page: ReturnType<typeof useTasksPage>;
  onNewTask: () => void;
}

/** What sits under the tabs: the list or board, or why there is neither. */
export function TasksBody({ page, onNewTask }: Props) {
  if (page.error && !page.hasData) {
    return (
      <ErrorCard
        context="tasks"
        httpError={{ message: "Your tasks could not be loaded." }}
        onRetry={() => void page.refetch()}
      />
    );
  }
  if (page.isLoading) return <TaskListSkeleton />;
  if (page.tasks.length === 0) {
    return (
      <TasksCard>
        <TasksEmpty
          tab={page.tab}
          isNewUser={page.allTasks.length === 0}
          onNewTask={onNewTask}
        />
      </TasksCard>
    );
  }
  if (page.view === "board") {
    return (
      <TaskBoard
        tasks={page.tasks}
        experts={page.experts}
        selectedID={page.taskID}
        onOpen={page.openTask}
      />
    );
  }
  return (
    <TaskList
      key={page.tab}
      tasks={page.tasks}
      experts={page.experts}
      selectedID={page.taskID}
      onOpen={page.openTask}
    />
  );
}
