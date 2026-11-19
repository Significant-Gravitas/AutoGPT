import {
  getListTasksQueryKey,
  useCreateTask,
  useRunTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import {
  buildCreateRequest,
  OTTO_VALUE,
  ownerChoices,
  resolveChoice,
} from "./helpers";

interface Args {
  experts: Expert[];
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
}

export function useNewTaskDialog({ experts, onOpenChange, onCreated }: Args) {
  const queryClient = useQueryClient();
  const [text, setText] = useState("");
  const [picked, setPicked] = useState(OTTO_VALUE);
  const createTask = useCreateTask({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() }),
    },
  });
  const runTask = useRunTask({
    mutation: {
      onSettled: () =>
        queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() }),
    },
  });
  const choices = ownerChoices(experts);
  const owner = resolveChoice(choices, picked);
  const isEmpty = text.trim().length === 0;
  const isSubmitting = createTask.isPending || runTask.isPending;
  const canSubmit = !isEmpty && !isSubmitting;

  function close() {
    onOpenChange(false);
    setText("");
  }

  /** Create the task and, unless it is saved for later, start it. */
  async function submit(e?: FormEvent, start = true) {
    e?.preventDefault();
    if (!canSubmit) return;
    let taskID: string;
    try {
      const res = await createTask.mutateAsync({
        data: buildCreateRequest(text, owner),
      });
      if (res.status !== 201) throw new Error("The task was not created.");
      taskID = res.data.id;
    } catch (error) {
      showError("Couldn't create the task", error);
      return;
    }
    const started = start && (await startTask(taskID));
    toast({
      title: started
        ? `Task started for ${owner.label}`
        : `Task saved for ${owner.label}`,
    });
    close();
    onCreated?.(taskID);
  }

  async function startTask(taskID: string) {
    try {
      await runTask.mutateAsync({ taskId: taskID, data: null });
      return true;
    } catch (error) {
      // The task exists either way; its tab can start it.
      showError("Created the task, but couldn't start it", error);
      return false;
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (!isKey(e, "Enter") || e.shiftKey) return;
    e.preventDefault();
    void submit();
  }

  return {
    text,
    setText,
    choices,
    owner,
    pickOwner: setPicked,
    canSubmit,
    isEmpty,
    isSubmitting,
    submit,
    saveForLater: () => void submit(undefined, false),
    onKeyDown,
    close,
  };
}

function showError(title: string, error: unknown) {
  toast({
    title,
    description: error instanceof Error ? error.message : undefined,
    variant: "destructive",
  });
}
