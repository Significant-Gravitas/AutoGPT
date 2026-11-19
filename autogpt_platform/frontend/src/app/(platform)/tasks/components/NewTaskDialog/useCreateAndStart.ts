import {
  getListTasksQueryKey,
  useCreateTask,
  useRunTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskCreateRequest } from "@/app/api/__generated__/models/taskCreateRequest";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { addTask, patchTask, replaceTask, restore } from "../../taskCache";
import { startedPatch } from "../../taskPatches";
import { standInTask } from "./helpers";
import { showStartFailed } from "./startFailedToast";

interface Created {
  task?: Task;
  refusal?: string;
}

const UNPROCESSABLE = 422;

function isRefusal(error: unknown): error is Error {
  return (
    error instanceof Error &&
    "status" in error &&
    error.status === UNPROCESSABLE
  );
}

/** Create a task, its card on the board at once, then start it; one
 *  message says how it went. */
export function useCreateAndStart() {
  const queryClient = useQueryClient();

  function rereadLists() {
    void queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
  }

  const createTask = useCreateTask({
    mutation: {
      onMutate: async ({ data }) => {
        const standIn = standInTask(data, Date.now());
        const snapshot = await addTask(queryClient, standIn);
        return { snapshot, standInID: standIn.id };
      },
      onSuccess: (res, _vars, context) => {
        if (!context) return;
        if (res.status === 201) {
          replaceTask(queryClient, context.standInID, res.data);
        } else {
          restore(queryClient, context.snapshot);
        }
      },
      onError: (_error, _vars, context) => {
        if (context) restore(queryClient, context.snapshot);
      },
      onSettled: rereadLists,
    },
  });
  const runTask = useRunTask({
    mutation: {
      onMutate: ({ taskId }) => patchTask(queryClient, taskId, startedPatch),
      onError: (_error, _vars, snapshot) => {
        if (snapshot) restore(queryClient, snapshot);
      },
      onSettled: rereadLists,
    },
  });

  /** The task the server made; or the server's sentence when it refused
   *  what was asked (422), for the dialog to show by its field; or neither,
   *  with a toast saying why. */
  async function create(request: TaskCreateRequest): Promise<Created> {
    try {
      const res = await createTask.mutateAsync({ data: request });
      if (res.status !== 201) throw new Error("The task was not created.");
      return { task: res.data };
    } catch (error) {
      if (isRefusal(error)) return { refusal: error.message };
      toast({
        title: "Couldn't create the task",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
      return {};
    }
  }

  async function start(taskID: string, owner: string) {
    try {
      await runTask.mutateAsync({ taskId: taskID, data: null });
      toast({ title: `Task started for ${owner}` });
    } catch (error) {
      // The task is saved either way; Retry starts it.
      showStartFailed({
        owner,
        error,
        onRetry: () => void start(taskID, owner),
      });
    }
  }

  return { create, start, isCreating: createTask.isPending };
}
