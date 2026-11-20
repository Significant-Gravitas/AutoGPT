import { useGetHomeDashboard } from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";
import { useRecheckWhileSchedulesUnknown } from "@/services/schedules/useRecheckWhileSchedulesUnknown";

const REFRESH_MS = 60_000;

export function useHomePage({ enabled = true }: { enabled?: boolean } = {}) {
  const recheckWhileSchedulesUnknown = useRecheckWhileSchedulesUnknown();
  const query = useGetHomeDashboard({
    query: {
      enabled,
      select: (response) => okData(response) ?? null,
      staleTime: 0,
      // Upcoming runs are unknown while the scheduler restarts; recheck after
      // its Retry-After rather than waiting out the minute, until the outage
      // outlasts a deploy.
      refetchInterval: (query) =>
        recheckWhileSchedulesUnknown(
          !!okData(query.state.data)?.schedules_unavailable,
        ) || REFRESH_MS,
      refetchOnWindowFocus: true,
    },
  });

  return {
    dashboard: query.data ?? null,
    isLoading: enabled && query.isLoading,
    isError: query.isError,
    isFetching: query.isFetching,
    refetch: query.refetch,
  };
}
