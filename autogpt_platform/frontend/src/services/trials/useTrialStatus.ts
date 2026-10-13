import {
  getGetTrialsGetTrialStatusQueryKey,
  type getTrialsGetTrialStatusResponse,
  useGetTrialsGetTrialStatus,
} from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

type TrialRefetchInterval =
  | number
  | ((trial: TrialStatusResponse | undefined) => number | false);

interface Options {
  poll?: boolean;
  refetchInterval?: TrialRefetchInterval;
}

export function useTrialStatus({ poll = false, refetchInterval }: Options = {}) {
  const userID = useAuthStore((state) => state.user?.id);
  return useGetTrialsGetTrialStatus({
    query: {
      queryKey: [...getGetTrialsGetTrialStatusQueryKey(), userID],
      enabled: Boolean(userID),
      retry: false,
      refetchInterval: (query) => {
        const trial = readTrial(query.state.data);
        if (typeof refetchInterval === "function") return refetchInterval(trial);
        if (refetchInterval !== undefined) return refetchInterval;
        return poll && trial?.active ? 30_000 : false;
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
