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
  refetchInterval?: TrialRefetchInterval;
}

export function useTrialStatus({ refetchInterval }: Options = {}) {
  const userID = useAuthStore((state) => state.user?.id);
  return useGetTrialsGetTrialStatus({
    query: {
      queryKey: [...getGetTrialsGetTrialStatusQueryKey(), userID],
      enabled: Boolean(userID),
      retry: false,
      refetchInterval:
        typeof refetchInterval === "function"
          ? (query) => refetchInterval(readTrial(query.state.data))
          : refetchInterval,
      select: readTrial,
    },
  });
}

function readTrial(response: getTrialsGetTrialStatusResponse | undefined) {
  return response?.status === 200 ? response.data : undefined;
}
