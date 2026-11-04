import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import { taskKey } from "../../data/helpers";
import type { TaskPriority } from "../../data/types";
import { useCreateTask } from "../../data/useTasksData";
import { ME_VALUE, toAssignee } from "./helpers";

interface Args {
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
}

export function useNewTaskDialog({ onOpenChange, onCreated }: Args) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [assignee, setAssignee] = useState(ME_VALUE);
  const [priority, setPriority] = useState<TaskPriority>("none");
  const [dueDate, setDueDate] = useState("");
  const createTask = useCreateTask();

  function reset() {
    setTitle("");
    setDescription("");
    setAssignee(ME_VALUE);
    setPriority("none");
    setDueDate("");
  }

  function close() {
    onOpenChange(false);
    reset();
  }

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const trimmed = title.trim();
    if (!trimmed || createTask.isPending) return;
    try {
      const task = await createTask.mutateAsync({
        title: trimmed,
        description: description.trim() || undefined,
        assignee: toAssignee(assignee),
        priority,
        due_date: dueDate || null,
      });
      toast({ title: `${taskKey(task.number)} created` });
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

  return {
    title,
    setTitle,
    description,
    setDescription,
    assignee,
    setAssignee,
    priority,
    setPriority,
    dueDate,
    setDueDate,
    canSubmit: title.trim().length > 0,
    isSubmitting: createTask.isPending,
    submit,
    close,
  };
}
