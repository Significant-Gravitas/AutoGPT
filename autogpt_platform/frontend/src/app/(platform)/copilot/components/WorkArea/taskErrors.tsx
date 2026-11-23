import { isTaskGone, isUnreachable } from "@/app/(platform)/tasks/taskErrors";
import { TASK_MUTATION_DEFAULTS } from "@/app/(platform)/tasks/taskMutations";
import { showGone, showUnreachable } from "@/app/(platform)/tasks/taskToasts";
import { toast } from "@/components/molecules/Toast/use-toast";
import { taskTabKey, useWorkAreaStore } from "../../workArea";

/** The panel's changes use the tasks' own options: sent now or not at all,
 *  so one made offline fails at once instead of replaying on reconnect. */
export const SEND_NOW = TASK_MUTATION_DEFAULTS;

const UUID =
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;

/** A server sentence fit to show: never an id. */
export function readableReason(error: unknown) {
  if (!(error instanceof Error) || !error.message) return undefined;
  const text = error.message
    .replace(UUID, "")
    .replace(/\s{2,}/g, " ")
    .trim();
  return text || undefined;
}

/** Close the task's tab and say once that it is gone: a read and a save
 *  can both find it gone, but only the first finds its tab still open. */
export function announceTaskGone(taskID: string) {
  const store = useWorkAreaStore.getState();
  const key = taskTabKey(taskID);
  if (!store.tabs.some((tab) => tab.key === key)) return;
  store.close(key);
  showGone();
}

interface FailureArgs {
  taskID: string;
  /** What did not happen, as the toast's title. */
  title: string;
  error: unknown;
  /** Send the same change again, for the offline toast's Retry. */
  retry: () => void;
}

/** One toast for a change the panel could not make: the task is gone, the
 *  server could not be reached (with Retry), or the server's own reason. */
export function showTaskFailure({ taskID, title, error, retry }: FailureArgs) {
  if (isTaskGone(error)) return announceTaskGone(taskID);
  if (isUnreachable(error)) return showUnreachable(retry);
  toast({
    title,
    description: readableReason(error),
    variant: "destructive",
  });
}
