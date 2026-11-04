import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import { useTaskAction } from "../../data/useTasksData";

export function useCancelTask(taskId: string) {
  const action = useTaskAction();
  const [isConfirming, setIsConfirming] = useState(false);

  function confirm() {
    if (action.isPending) return;
    action.mutate(
      { id: taskId, action: "stop" },
      {
        onSuccess: () => setIsConfirming(false),
        onError: (error) =>
          toast({
            title: "Couldn't cancel the task",
            description: error instanceof Error ? error.message : undefined,
            variant: "destructive",
          }),
      },
    );
  }

  return {
    isConfirming,
    ask: () => setIsConfirming(true),
    keep: () => setIsConfirming(false),
    confirm,
    isCancelling: action.isPending,
  };
}
