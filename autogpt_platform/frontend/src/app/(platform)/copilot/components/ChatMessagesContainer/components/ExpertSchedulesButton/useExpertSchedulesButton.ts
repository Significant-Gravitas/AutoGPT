import { useGetExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { getExpertSchedules } from "@/app/(platform)/team/helpers";
import { useState } from "react";

export function useExpertSchedulesButton(expertId: string) {
  const [isOpen, setIsOpen] = useState(false);

  const expertQuery = useGetExpert(expertId, {
    query: { select: (res) => okData(res) ?? null },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [] },
  });

  const expert = expertQuery.data ?? null;
  const allSchedules = schedulesQuery.data ?? [];
  // Until the expert record arrives, the expert_id stamp on schedules is
  // enough to keep the count honest; the workflow-ref join in
  // getExpertSchedules only adds legacy schedules created before stamping.
  const schedules = expert
    ? getExpertSchedules(expert, allSchedules)
    : allSchedules.filter((schedule) => schedule.expert_id === expertId);

  return { isOpen, setIsOpen, schedules };
}
