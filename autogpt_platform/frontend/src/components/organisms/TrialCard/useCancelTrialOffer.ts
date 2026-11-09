import { useGetBrainDumpSuggestedTasks } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import {
  getTaskState,
  TOP_TASKS,
} from "@/app/(platform)/copilot/components/SuggestedTasks/helpers";
import { useFeedbackPrompt } from "@/components/organisms/FeedbackPrompts/useFeedbackPrompt";
import { SUGGESTED_TASK_PARAM } from "@/app/(platform)/copilot/useWorkflowImportAutoSubmit";
import { useRouter } from "next/navigation";
import { useState } from "react";

// Home marks the row sent once the chat exists (useWorkflowImportAutoSubmit).
function getTaskHref(task: SuggestedTask) {
  const params = new URLSearchParams({ autosubmit: "true" });
  if (task.expert_id) params.set("expertId", task.expert_id);
  params.set(SUGGESTED_TASK_PARAM, task.id);
  return `/home?${params}#prompt=${encodeURIComponent(task.prompt)}`;
}

// The trial cancel prompt: what Otto can offer instead, and where the answers
// go.
export function useCancelTrialOffer() {
  const router = useRouter();
  const prompt = useFeedbackPrompt("trial_cancel");
  const [hasOpened, setHasOpened] = useState(false);
  const { data: response } = useGetBrainDumpSuggestedTasks({
    query: { enabled: hasOpened },
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

  function startTask(title: string) {
    const task = tasks.find((candidate) => candidate.title === title);
    if (!task) return;
    prompt.save({ offer_taken: true });
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
