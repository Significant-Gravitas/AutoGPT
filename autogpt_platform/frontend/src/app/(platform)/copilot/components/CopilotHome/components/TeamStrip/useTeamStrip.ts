import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export function useTeamStrip() {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  const expertsQuery = useListExperts({
    query: {
      // okData, like every sibling query here: an error body must never
      // reach the render layer typed as Expert[].
      select: (res) => okData(res) ?? [],
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
