import {
  getListTasksQueryKey,
  useCreateTask,
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
  const choices = ownerChoices(experts);
  const owner = resolveChoice(choices, picked);
  const isEmpty = text.trim().length === 0;
  const canSubmit = !isEmpty && !createTask.isPending;

  function close() {
    onOpenChange(false);
    setText("");
  }

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    if (!canSubmit) return;
    try {
      const res = await createTask.mutateAsync({
        data: buildCreateRequest(text, owner),
      });
      if (res.status !== 201) throw new Error("The task was not created.");
      toast({ title: `Task created for ${owner.label}` });
      close();
      onCreated?.(res.data.id);
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
    owner,
    pickOwner: setPicked,
    canSubmit,
    isEmpty,
    isSubmitting: createTask.isPending,
    submit,
    onKeyDown,
    close,
  };
}
