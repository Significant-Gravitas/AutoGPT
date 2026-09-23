import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import {
  getGetTrialsGetTrialStatusQueryKey,
  type getTrialsGetTrialStatusResponseSuccess,
} from "@/app/api/__generated__/endpoints/trials/trials";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { QueryClient } from "@tanstack/react-query";

export async function updateTrialStatusCache({
  queryClient,
  userID,
  response,
}: {
  queryClient: QueryClient;
  userID: string;
  response: getTrialsGetTrialStatusResponseSuccess;
}) {
  const queryKey = [...getGetTrialsGetTrialStatusQueryKey(), userID];
  await queryClient.cancelQueries({ queryKey, exact: true });
  if (useAuthStore.getState().user?.id !== userID) return false;
  queryClient.setQueryData(queryKey, response);
  await queryClient.invalidateQueries({
    queryKey: getGetSubscriptionStatusQueryKey(),
  });
  return useAuthStore.getState().user?.id === userID;
}
