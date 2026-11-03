import { useState } from "react";
import { useCreateTask } from "../../../../data/useTasksData";

export function useSubtaskList(parentId: string) {
  const [isAdding, setIsAdding] = useState(false);
  const [title, setTitle] = useState("");
  const createTask = useCreateTask();

  function submit() {
    const trimmed = title.trim();
    if (!trimmed || createTask.isPending) return;
    createTask.mutate(
      { title: trimmed, parent_id: parentId },
      { onSuccess: () => setTitle("") },
    );
  }

  function cancel() {
    setTitle("");
    setIsAdding(false);
  }

  return {
    isAdding,
    startAdding: () => setIsAdding(true),
    cancel,
    title,
    setTitle,
    submit,
    isCreating: createTask.isPending,
    isError: createTask.isError,
  };
}
