import { QueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

import {
  getGetTrialsGetTrialStatusQueryKey,
  getTrialsGetTrialStatus,
} from "@/app/api/__generated__/endpoints/trials/trials";
import {
  getGetSubscriptionStatusQueryKey,
  getSubscriptionStatus,
} from "@/app/api/__generated__/endpoints/credits/credits";
import {
  getGetV2GetCopilotUsageQueryKey,
  getV2GetCopilotUsage,
} from "@/app/api/__generated__/endpoints/chat/chat";

export async function refreshActivation(client: QueryClient, userID: string) {
  const keys = [
    getGetTrialsGetTrialStatusQueryKey(),
    getGetSubscriptionStatusQueryKey(),
    getGetV2GetCopilotUsageQueryKey(),
  ];
  await Promise.all(keys.map((queryKey) => client.cancelQueries({ queryKey })));
  await Promise.all(
    keys.map((queryKey) =>
      client.invalidateQueries({ queryKey, refetchType: "none" }),
    ),
  );
  const [trial, subscription, usage] = await Promise.all([
    client.fetchQuery({
      queryKey: [...keys[0], userID],
      queryFn: ({ signal }) =>
        ownedResponse(() => getTrialsGetTrialStatus({ signal }), userID),
      staleTime: 0,
    }),
    client.fetchQuery({
      queryKey: keys[1],
      queryFn: ({ signal }) =>
        ownedResponse(() => getSubscriptionStatus({ signal })),
      staleTime: 0,
    }),
    client.fetchQuery({
      queryKey: keys[2],
      queryFn: ({ signal }) =>
        ownedResponse(() => getV2GetCopilotUsage({ signal })),
      staleTime: 0,
    }),
  ]);
  if (
    trial.status !== 200 ||
    subscription.status !== 200 ||
    usage.status !== 200 ||
    trial.data.active ||
    subscription.data.tier !== "PRO" ||
    usage.data.tier !== "PRO"
  )
    throw new Error("Your plan and allowance are still being refreshed");
}

async function ownedResponse<T>(
  request: () => Promise<T>,
  expectedUserID?: string,
) {
  const userID = expectedUserID ?? useAuthStore.getState().user?.id;
  if (!userID || useAuthStore.getState().user?.id !== userID)
    throw new Error("Account changed during activation");
  const result = await request();
  if (useAuthStore.getState().user?.id !== userID)
    throw new Error("Account changed during activation");
  return result;
}
