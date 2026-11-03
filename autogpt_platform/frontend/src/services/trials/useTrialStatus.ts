import {
  getGetTrialsGetTrialStatusQueryKey,
  type getTrialsGetTrialStatusResponse,
  useGetTrialsGetTrialStatus,
} from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

interface Options {
  poll?: boolean;
  refetchInterval?: (trial: TrialStatusResponse | undefined) => number | false;
}

export function useTrialStatus({
  poll = false,
  refetchInterval,
}: Options = {}) {
  const userID = useAuthStore((state) => state.user?.id);
  return useGetTrialsGetTrialStatus({
    query: {
      queryKey: [...getGetTrialsGetTrialStatusQueryKey(), userID],
      enabled: Boolean(userID),
      retry: false,
      refetchInterval: (query) => {
        if (refetchInterval)
          return refetchInterval(readTrial(query.state.data));
        const response = query.state.data;
        return poll && response?.status === 200 && response.data.active
          ? 30_000
          : false;
      },
      refetchIntervalInBackground: false,
      refetchOnWindowFocus: true,
      staleTime: 10_000,
      select: readTrial,
    },
  });
}

function readTrial(response: getTrialsGetTrialStatusResponse | undefined) {
  return response?.status === 200 ? response.data : undefined;
}
