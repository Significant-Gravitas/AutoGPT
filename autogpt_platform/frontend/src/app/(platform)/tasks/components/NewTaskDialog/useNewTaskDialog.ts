import { toast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { Key, storage } from "@/services/storage/local-storage";
import { useState, type KeyboardEvent } from "react";
import { taskKey } from "../../data/helpers";
import type { ExpertOption } from "../../data/types";
import { useCreateTask } from "../../data/useTasksData";
import {
  assigneeChoices,
  createdMessage,
  resolveChoice,
  splitTaskText,
  toAssignee,
} from "./helpers";

interface Args {
  experts: ExpertOption[];
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
}

export function useNewTaskDialog({ experts, onOpenChange, onCreated }: Args) {
  const [text, setText] = useState("");
  const [picked, setPicked] = useState(() =>
    storage.get(Key.NEW_TASK_ASSIGNEE),
  );
  const createTask = useCreateTask();
  const choices = assigneeChoices(experts);
  const assignee = resolveChoice(choices, picked);
  const isEmpty = text.trim().length === 0;
  const canSubmit = !isEmpty && !createTask.isPending;

  function pickAssignee(value: string) {
    setPicked(value);
    storage.set(Key.NEW_TASK_ASSIGNEE, value);
  }

  function close() {
    onOpenChange(false);
    setText("");
  }

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!canSubmit) return;
    try {
      const task = await createTask.mutateAsync({
        ...splitTaskText(text),
        assignee: toAssignee(assignee.value),
      });
      toast({ title: createdMessage(taskKey(task.number), assignee) });
      close();
      onCreated?.(task.id);
    } catch (error) {
      toast({
        title: "Couldn't create the task",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
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
    assignee,
    pickAssignee,
    canSubmit,
    isEmpty,
    isSubmitting: createTask.isPending,
    submit,
    onKeyDown,
    close,
  };
}
