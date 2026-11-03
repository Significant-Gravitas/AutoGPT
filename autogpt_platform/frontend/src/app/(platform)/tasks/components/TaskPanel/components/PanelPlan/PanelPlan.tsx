"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { PlanCard } from "@/app/(platform)/copilot/tools/TaskPlan/TaskPlan";
import { useTaskPlan } from "../../../../data/useTasksData";

interface Props {
  planId: string;
}

/** Otto's plan behind this task, the same card the chat shows. */
export function PanelPlan({ planId }: Props) {
  const { data: plan, isLoading } = useTaskPlan(planId);

  if (isLoading) {
    return (
      <div className="px-5 py-4">
        <Skeleton className="h-28 w-full rounded-2xl" />
      </div>
    );
  }
  if (!plan) return null;

  return (
    <section aria-label="Plan" className="animate-inbox-enter px-5 py-2">
      <PlanCard plan={plan} />
    </section>
  );
}
