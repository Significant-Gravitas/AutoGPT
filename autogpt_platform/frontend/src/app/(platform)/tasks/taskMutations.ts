import type { Task } from "@/app/api/__generated__/models/task";
import type { QueryClient } from "@tanstack/react-query";
import {
  cachedTask,
  patchTask,
  restore,
  type CacheSnapshot,
  type TaskPatch,
} from "./taskCache";
import { isTaskGone, isUnreachable } from "./taskErrors";
import { archivedPatch } from "./taskPatches";
import { announceTaskChange } from "./taskSync";
import { showError, showGone, showUnreachable } from "./taskToasts";

/** Options every task change uses. It runs at once, online or not: one
 *  clicked offline fails now and is rolled back, never paused and sent
 *  minutes later. No retry: a change is sent once. */
export const TASK_MUTATION_DEFAULTS = {
  networkMode: "always",
  retry: false,
} as const;

interface TaskContext {
  before: Task | null;
  snapshot: CacheSnapshot;
}

interface Args<TVars> {
  queryClient: QueryClient;
  taskID: string;
  patchOf: (vars: TVars) => TaskPatch;
  errorTitle: (vars: TVars) => string;
  /** Sends the same change again, for the offline toast's Retry. */
  retry: (vars: TVars) => void;
  /** Re-read the task and the lists, however it went. */
  onSettled: () => void;
  /** An error that means the change is already true (a second start). */
  isDone?: (error: unknown, before: Task | null) => boolean;
  /** After the server took it, with the task as it was before. */
  afterSuccess?: (before: Task | null) => void;
}

/** A task change shown at once in every cache. On error it is put back
 *  with one toast: Retry when the server was out of reach, "no longer
 *  exists" (and the card leaves) on a 404, the server's sentence
 *  otherwise. A success is announced to the app's other tabs. */
export function optimisticTask<TVars>(args: Args<TVars>) {
  const { queryClient, taskID } = args;
  return {
    ...TASK_MUTATION_DEFAULTS,
    onMutate: async (vars: TVars): Promise<TaskContext> => ({
      before: cachedTask(queryClient, taskID),
      snapshot: await patchTask(queryClient, taskID, args.patchOf(vars)),
    }),
    onError: (
      error: unknown,
      vars: TVars,
      context: TaskContext | undefined,
    ) => {
      if (args.isDone?.(error, context?.before ?? null)) return;
      if (context) restore(queryClient, context.snapshot);
      if (isTaskGone(error)) {
        void patchTask(queryClient, taskID, archivedPatch);
        showGone();
        return;
      }
      if (isUnreachable(error)) {
        showUnreachable(() => args.retry(vars));
        return;
      }
      showError(args.errorTitle(vars), error);
    },
    onSuccess: (
      _data: unknown,
      _vars: TVars,
      context: TaskContext | undefined,
    ) => {
      announceTaskChange(taskID);
      args.afterSuccess?.(context?.before ?? null);
    },
    onSettled: args.onSettled,
  };
}
