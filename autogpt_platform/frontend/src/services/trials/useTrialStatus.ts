import {
  getGetTrialsGetTrialStatusQueryKey,
  useGetTrialsGetTrialStatus,
} from "@/app/api/__generated__/endpoints/trials/trials";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

interface Options {
  refetchInterval?: number;
}

export function useTrialStatus({ refetchInterval }: Options = {}) {
  const userID = useAuthStore((state) => state.user?.id);
  return useGetTrialsGetTrialStatus({
    query: {
      queryKey: [...getGetTrialsGetTrialStatusQueryKey(), userID],
      enabled: Boolean(userID),
      retry: false,
      refetchInterval,
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });
}
