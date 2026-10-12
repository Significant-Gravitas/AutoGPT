import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import {
  getGetTrialsGetTrialStatusQueryKey,
  type getTrialsGetTrialStatusResponseSuccess,
} from "@/app/api/__generated__/endpoints/trials/trials";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { QueryClient } from "@tanstack/react-query";

// onApplied runs as soon as the trial status is written, before the slower
// plan status refresh, so UI tied to the new trial state shows with it.
export async function updateTrialStatusCache({
  queryClient,
  userID,
  response,
  onApplied,
}: {
  queryClient: QueryClient;
  userID: string;
  response: getTrialsGetTrialStatusResponseSuccess;
  onApplied?: () => void;
}) {
  const queryKey = [...getGetTrialsGetTrialStatusQueryKey(), userID];
  await queryClient.cancelQueries({ queryKey, exact: true });
  if (useAuthStore.getState().user?.id !== userID) return false;
  queryClient.setQueryData(queryKey, response);
  onApplied?.();
  await queryClient.invalidateQueries({
    queryKey: getGetSubscriptionStatusQueryKey(),
  });
  return useAuthStore.getState().user?.id === userID;
}
