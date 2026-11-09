import {
  getGetBrainDumpSuggestedTasksQueryKey,
  type getBrainDumpSuggestedTasksResponse,
  useGetBrainDumpSuggestedTasks,
  useUpdateBrainDumpSuggestedTasks,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import {
  getTaskState,
  TOP_TASKS,
  withState,
} from "@/app/(platform)/copilot/components/SuggestedTasks/helpers";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";

const QUERY_KEY = getGetBrainDumpSuggestedTasksQueryKey();

function getTaskHref(task: SuggestedTask) {
  const params = new URLSearchParams({ autosubmit: "true" });
  if (task.expert_id) params.set("expertId", task.expert_id);
  return `/home?${params}#prompt=${encodeURIComponent(task.prompt)}`;
}

// The trial cancel prompt: what Otto can offer instead, and where the answers
// go. A failed write here must not surface, so the task list's own toasts are
// left out on purpose.
export function useCancelTrialOffer() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const prompt = useFeedbackPrompt("trial_cancel");
  const [hasOpened, setHasOpened] = useState(false);
  const { data: response } = useGetBrainDumpSuggestedTasks({
    query: { enabled: hasOpened },
  });
  const { mutate } = useUpdateBrainDumpSuggestedTasks({
    mutation: {
      scope: { id: "brain-dump-suggested-tasks" },
      retry: 3,
      onSettled: () =>
        void queryClient.invalidateQueries({ queryKey: QUERY_KEY }),
    },
  });

  const list = response?.status === 200 ? response.data : null;
  const openTasks = list?.stored
    ? (list.tasks ?? []).filter((task) => getTaskState(task) === "open")
    : [];
  const tasks = openTasks.slice(0, TOP_TASKS);

  function handleOpen() {
    setHasOpened(true);
    void prompt.claim();
  }

  function markSent(task: SuggestedTask) {
    queryClient.setQueryData<getBrainDumpSuggestedTasksResponse>(
      QUERY_KEY,
      (current) => {
        if (!current || current.status !== 200) return current;
        return {
          ...current,
          data: {
            ...current.data,
            tasks: withState({
              tasks: current.data.tasks ?? [],
              ids: [task.id],
              state: "sent",
            }),
          },
        };
      },
    );
    mutate({ data: { task_ids: [task.id], state: "sent" } });
  }

  function startTask(title: string) {
    const task = tasks.find((candidate) => candidate.title === title);
    if (!task) return;
    prompt.save({ offer_taken: true });
    markSent(task);
    router.push(getTaskHref(task));
  }

  function tryOtto() {
    prompt.save({ offer_taken: true });
    router.push("/home");
  }

  return {
    starters: tasks.map((task) => task.title),
    handleOpen,
    save: prompt.save,
    startTask,
    tryOtto,
  };
}
