import { useApproveTaskPlan } from "@/app/(platform)/tasks/data/useTasksData";
import type { TaskPlan } from "@/app/(platform)/tasks/data/types";
import { toast } from "@/components/molecules/Toast/use-toast";

export function useTaskPlanCard(plan: TaskPlan) {
  const approve = useApproveTaskPlan();
  const started = approve.data ?? null;
  const alreadyStarted = plan.status === "approved";
  const parentId = started?.id ?? plan.task_id ?? null;

  function handleApprove() {
    approve.mutate(plan, {
      onError: () =>
        toast({
          title: "Couldn't start the plan",
          description: "Please try again in a moment.",
          variant: "destructive",
        }),
    });
  }

  return {
    handleApprove,
    isApproving: approve.isPending,
    isStarted: alreadyStarted || !!started,
    isCancelled: plan.status === "rejected",
    parentId,
    parentNumber: started?.number ?? null,
  };
}
