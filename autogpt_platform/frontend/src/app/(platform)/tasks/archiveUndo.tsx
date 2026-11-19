import {
  getListTasksQueryKey,
  unarchiveTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { QueryClient } from "@tanstack/react-query";
import { addTask, replaceTask, restore } from "./taskCache";
import { isUnreachable } from "./taskErrors";
import { announceTaskChange } from "./taskSync";
import { showError, showUnreachable } from "./taskToasts";

/** How long Undo stays on offer. */
const UNDO_MS = 8_000;

/** Bring an archived task back: its card returns at once, and goes again
 *  with a toast if the server says no. Called from a toast, after the
 *  card (and its menu) have gone, so it uses the query client directly. */
export async function undoArchive(queryClient: QueryClient, task: Task) {
  const snapshot = await addTask(queryClient, task);
  try {
    const res = await unarchiveTask(task.id);
    if (res.status !== 200) throw new Error("The task could not come back.");
    replaceTask(queryClient, task.id, res.data);
    announceTaskChange(task.id);
  } catch (error) {
    restore(queryClient, snapshot);
    if (isUnreachable(error)) {
      showUnreachable(() => void undoArchive(queryClient, task));
    } else {
      showError("Couldn't bring the task back", error);
    }
  } finally {
    void queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
  }
}

/** "Archived", with Undo for a few seconds when the task can come back. */
export function showArchived(onUndo?: () => void) {
  toast({
    title: "Archived",
    duration: UNDO_MS,
    action: onUndo ? (
      <Button type="button" variant="secondary" size="xs" onClick={onUndo}>
        Undo
      </Button>
    ) : undefined,
  });
}
