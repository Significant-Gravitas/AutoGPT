import { useListCopilotFollowupSchedules } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";

export function useFollowupsPage() {
  const query = useListCopilotFollowupSchedules({
    query: {
      select: (res) => okData(res) ?? [],
    },
  });

  return {
    followups: query.data ?? [],
    isLoading: query.isLoading,
    error: query.error,
    refetchFollowups: query.refetch,
  };
}
