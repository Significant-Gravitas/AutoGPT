import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export function useTeamStrip() {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  const expertsQuery = useListExperts({
    query: {
      select: (res) => res.data as Expert[],
      enabled: isExpertsEnabled,
    },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled: isExpertsEnabled },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  return {
    isVisible: isExpertsEnabled && hiredExperts.length > 0,
    hiredExperts,
    schedules: schedulesQuery.data ?? [],
  };
}
