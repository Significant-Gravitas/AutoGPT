import {
  getGetTaskQueryKey,
  getGetTaskRecordQueryKey,
  getListTasksQueryKey,
  useUpdateTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { useQueryClient, type QueryKey } from "@tanstack/react-query";
import { announceTaskChange } from "@/app/(platform)/tasks/taskSync";
import { SEND_NOW, showTaskFailure } from "../taskErrors";

/** The task's own words a person may change in its panel. */
export type TaskTextField = "description" | "goal" | "done_when";

const SAVE_ERRORS: Record<TaskTextField, string> = {
  description: "Couldn't save the description",
  goal: "Couldn't save the goal",
  done_when: "Couldn't save when it's done",
};

interface Cached {
  status: number;
  data: unknown;
}

/** Put the new words on the cached task (its record, and the task on its
 *  own) at once; hand back what was there to put back on failure. */
function patchCached(
  queryClient: ReturnType<typeof useQueryClient>,
  key: QueryKey,
  field: TaskTextField,
  text: string,
) {
  const before = queryClient.getQueryData<Cached>(key);
  if (before?.status !== 200 || !before.data) return null;
  const data = before.data as Record<string, unknown>;
  const next =
    "task" in data && data.task && typeof data.task === "object"
      ? { ...data, task: { ...data.task, [field]: text } }
      : { ...data, [field]: text };
  queryClient.setQueryData(key, { ...before, data: next });
  return before;
}

/** Save a task's description, goal or "done when" through PATCH, showing
 *  the new words straight away. It is sent now or not at all: offline, the
 *  old words come back at once with a Retry. */
export function useTaskFieldSave(taskID: string) {
  const queryClient = useQueryClient();
  const keys = [getGetTaskRecordQueryKey(taskID), getGetTaskQueryKey(taskID)];

  const update = useUpdateTask({
    mutation: {
      ...SEND_NOW,
      onMutate: async ({ data }) => {
        const field = (Object.keys(data) as TaskTextField[])[0];
        const text = String(data[field] ?? "");
        await Promise.all(
          keys.map((queryKey) => queryClient.cancelQueries({ queryKey })),
        );
        return keys.map((key) => ({
          key,
          before: patchCached(queryClient, key, field, text),
        }));
      },
      onError: (error, vars, saved) => {
        for (const { key, before } of saved ?? []) {
          if (before) queryClient.setQueryData(key, before);
        }
        const field = (Object.keys(vars.data) as TaskTextField[])[0];
        showTaskFailure({
          taskID,
          title: SAVE_ERRORS[field] ?? "Couldn't save the task",
          error,
          retry: () => update.mutate(vars),
        });
      },
      onSuccess: () => announceTaskChange(taskID),
      onSettled: () => {
        void queryClient.invalidateQueries({
          queryKey: getListTasksQueryKey(),
        });
        for (const queryKey of keys) {
          void queryClient.invalidateQueries({ queryKey });
        }
      },
    },
  });

  function save(field: TaskTextField, text: string) {
    update.mutate({ taskId: taskID, data: { [field]: text } });
  }

  return { save };
}
