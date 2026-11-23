import { patchTask, restore } from "@/app/(platform)/tasks/taskCache";
import { announceTaskChange } from "@/app/(platform)/tasks/taskSync";
import {
  getGetTaskQueryKey,
  getGetTaskRecordQueryKey,
  getListTasksQueryKey,
  useReviewTask,
  type getTaskRecordResponse,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { useQueryClient } from "@tanstack/react-query";
import { SEND_NOW, showTaskFailure } from "../taskErrors";

/** Accept a finished task's result, or ask for changes, which runs it
 *  again with the note as what to change. Accepting shows at once and is
 *  put back with the server's reason if it says no. */
export function useTaskReview(taskID: string) {
  const queryClient = useQueryClient();
  const recordKey = getGetTaskRecordQueryKey(taskID);

  function refresh() {
    void queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
    void queryClient.invalidateQueries({
      queryKey: getGetTaskQueryKey(taskID),
    });
    void queryClient.invalidateQueries({ queryKey: recordKey });
  }

  function markVerified() {
    queryClient.setQueryData<getTaskRecordResponse>(recordKey, (old) =>
      old?.status === 200
        ? { ...old, data: { ...old.data, is_unverified: false } }
        : old,
    );
  }

  const accept = useReviewTask({
    mutation: {
      ...SEND_NOW,
      onMutate: async () => {
        const snapshot = await patchTask(queryClient, taskID, (task) => ({
          ...task,
          review: { state: "accepted", note: null, at: new Date() },
        }));
        markVerified();
        return snapshot;
      },
      onError: (error, vars, snapshot) => {
        if (snapshot) restore(queryClient, snapshot);
        showTaskFailure({
          taskID,
          title: "Couldn't accept the result",
          error,
          retry: () => accept.mutate(vars),
        });
      },
      onSuccess: () => announceTaskChange(taskID),
      onSettled: refresh,
    },
  });

  const requestChanges = useReviewTask({
    mutation: {
      ...SEND_NOW,
      onSuccess: async (response) => {
        if (response.status !== 200) return;
        const started: Task = response.data;
        await patchTask(queryClient, taskID, () => started);
        announceTaskChange(taskID);
      },
      onError: (error, vars) =>
        showTaskFailure({
          taskID,
          title: "Couldn't ask for changes",
          error,
          retry: () => requestChanges.mutate(vars),
        }),
      onSettled: refresh,
    },
  });

  return {
    accept: () =>
      accept.mutate({ taskId: taskID, data: { decision: "accept" } }),
    requestChanges: (note: string, onSent: () => void) =>
      requestChanges.mutate(
        { taskId: taskID, data: { decision: "request_changes", note } },
        { onSuccess: onSent },
      ),
    isAccepting: accept.isPending,
    isRequesting: requestChanges.isPending,
  };
}
