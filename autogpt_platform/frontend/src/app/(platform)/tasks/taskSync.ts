import {
  getGetTaskQueryKey,
  getGetTaskRecordQueryKey,
  getListTasksQueryKey,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

const CHANNEL = "tasks";

interface TaskChange {
  taskID: string | null;
}

function channel() {
  return typeof BroadcastChannel === "undefined"
    ? null
    : new BroadcastChannel(CHANNEL);
}

/** Tell the app's other tabs a task changed, so they re-read it. */
export function announceTaskChange(taskID: string | null) {
  const out = channel();
  if (!out) return;
  const message: TaskChange = { taskID };
  out.postMessage(message);
  out.close();
}

/** Re-read the task lists, and the task named, when another tab changes
 *  one. Mount once where tasks are shown. */
export function useTaskSync() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const incoming = channel();
    if (!incoming) return;
    function receive(event: MessageEvent<TaskChange>) {
      void queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
      const id = event.data?.taskID;
      if (!id) return;
      void queryClient.invalidateQueries({ queryKey: getGetTaskQueryKey(id) });
      void queryClient.invalidateQueries({
        queryKey: getGetTaskRecordQueryKey(id),
      });
    }
    incoming.addEventListener("message", receive);
    return () => {
      incoming.removeEventListener("message", receive);
      incoming.close();
    };
  }, [queryClient]);
}
