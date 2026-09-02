import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetTaskQueryKey,
  getListTasksQueryKey,
  useAcceptTask,
  useRejectTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";

export function useTaskOutcomeReview(taskId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isRequestingChanges, setIsRequestingChanges] = useState(false);
  const [note, setNote] = useState("");

  async function refreshTask() {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: getGetTaskQueryKey(taskId) }),
      queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() }),
    ]);
  }

  const { mutate: accept, isPending: isAccepting } = useAcceptTask({
    mutation: {
      onSuccess: async (res) => {
        toast({ title: okData(res)?.message ?? "Outcome accepted" });
        await refreshTask();
      },
      onError: () =>
        toast({
          title: "Could not accept the outcome",
          description: "Please try again.",
          variant: "destructive",
        }),
    },
  });

  const { mutate: reject, isPending: isRejecting } = useRejectTask({
    mutation: {
      onSuccess: async (res) => {
        toast({ title: okData(res)?.message ?? "Changes requested" });
        setIsRequestingChanges(false);
        setNote("");
        await refreshTask();
      },
      onError: () =>
        toast({
          title: "Could not send your changes",
          description: "Please try again.",
          variant: "destructive",
        }),
    },
  });

  function handleAccept() {
    accept({ taskId });
  }

  function revealNote() {
    setIsRequestingChanges(true);
  }

  function cancelNote() {
    setIsRequestingChanges(false);
    setNote("");
  }

  function submitChanges() {
    const trimmed = note.trim();
    if (!trimmed || isRejecting) return;
    reject({ taskId, data: { note: trimmed } });
  }

  return {
    isRequestingChanges,
    note,
    setNote,
    handleAccept,
    revealNote,
    cancelNote,
    submitChanges,
    isAccepting,
    isRejecting,
  };
}
