import { toast } from "@/components/molecules/Toast/use-toast";
import { useTaskAction } from "../../data/useTasksData";

export function useInboxBoard() {
  const action = useTaskAction();

  function run(id: string, name: string, failure: string) {
    action.mutate(
      { id, action: name },
      {
        onError: (error) =>
          toast({
            title: failure,
            description: error instanceof Error ? error.message : undefined,
            variant: "destructive",
          }),
      },
    );
  }

  return {
    cancel(id: string) {
      if (!window.confirm("Stop this task? The expert stops working on it."))
        return;
      run(id, "stop", "Couldn't cancel the task");
    },
    retry: (id: string) =>
      run(id, "retry_failed_part", "Couldn't retry the task"),
  };
}
