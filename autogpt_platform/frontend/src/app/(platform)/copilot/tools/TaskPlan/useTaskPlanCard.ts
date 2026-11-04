import { useQueryClient } from "@tanstack/react-query";
import { useContext, useState } from "react";
import { useConfirmTaskPlan } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { toTask } from "@/app/(platform)/tasks/data/normalize";
import {
  invalidateTasks,
  useApproveTaskPlan,
} from "@/app/(platform)/tasks/data/useTasksData";
import type { TaskPlan, TaskPlanStep } from "@/app/(platform)/tasks/data/types";
import { toast } from "@/components/molecules/Toast/use-toast";
import { stepsChanged, toConfirmSteps } from "./helpers";
import { LatestPlanContext } from "./latestPlan";

export function useTaskPlanCard(plan: TaskPlan) {
  const queryClient = useQueryClient();
  const approve = useApproveTaskPlan();
  const confirmEdited = useConfirmTaskPlan();
  const latestPlanId = useContext(LatestPlanContext);
  const [draft, setDraft] = useState<TaskPlanStep[] | null>(null);

  const editedParent =
    confirmEdited.data?.status === 200
      ? toTask(confirmEdited.data.data.parent_task)
      : null;
  const started = approve.data ?? editedParent;
  const isStarted = plan.status === "approved" || !!started;
  const isCancelled = plan.status === "rejected";
  const isSuperseded =
    !isStarted && latestPlanId !== null && latestPlanId !== plan.id;
  const isApproving = approve.isPending || confirmEdited.isPending;
  const steps = draft ?? plan.steps;
  const canAct = !isStarted && !isCancelled && !isSuperseded;
  const hasSteps = toConfirmSteps(steps).length > 0;

  function handleError() {
    toast({
      title: "Couldn't start the plan",
      description: "Please try again in a moment.",
      variant: "destructive",
    });
  }

  function handleApprove() {
    if (!canAct || !hasSteps) return;
    if (draft && stepsChanged(plan.steps, draft)) {
      confirmEdited.mutate(
        { planId: plan.id, data: { steps: toConfirmSteps(draft) } },
        {
          onSuccess: () => invalidateTasks(queryClient),
          onError: handleError,
        },
      );
    } else {
      approve.mutate(plan, { onError: handleError });
    }
  }

  function handleEdit() {
    setDraft(plan.steps);
  }

  function handleCancelEdit() {
    setDraft(null);
  }

  function handleStepTitle(index: number, title: string) {
    setDraft((current) =>
      (current ?? plan.steps).map((step, i) =>
        i === index ? { ...step, title } : step,
      ),
    );
  }

  function handleRemoveStep(index: number) {
    setDraft((current) =>
      (current ?? plan.steps).filter((_, i) => i !== index),
    );
  }

  return {
    steps,
    isEditing: draft !== null && canAct && !isApproving,
    canAct,
    hasSteps,
    handleApprove,
    handleEdit,
    handleCancelEdit,
    handleStepTitle,
    handleRemoveStep,
    isApproving,
    isStarted,
    isCancelled,
    isSuperseded,
    parentId: started?.id ?? plan.task_id ?? null,
    parentNumber: started?.number ?? null,
  };
}
