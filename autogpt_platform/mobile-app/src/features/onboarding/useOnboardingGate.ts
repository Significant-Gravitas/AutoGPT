import { skipToken, useQuery } from "@tanstack/react-query";

import { getOnboardingCompleted, onboardingKeys } from "@/lib/api/onboarding";

import { GATE_TIMEOUT_MS, gateState } from "./gate";
import { markOnboarded, useIsOnboarded } from "./onboardedStore";

/** Asks the server once per account whether onboarding is finished, and remembers a yes. */
export function useOnboardingGate(userID: string | undefined) {
  const isRemembered = useIsOnboarded(userID);
  const query = useQuery({
    queryKey: [...onboardingKeys.completed, userID],
    queryFn: userID && !isRemembered ? () => askServer(userID) : skipToken,
    retry: false,
    staleTime: Infinity,
  });
  return gateState({
    isRemembered,
    status: query.status,
    isCompleted: query.data?.is_completed,
  });
}

async function askServer(userID: string) {
  const answer = await withTimeout(getOnboardingCompleted(), GATE_TIMEOUT_MS);
  if (answer.is_completed) markOnboarded(userID);
  return answer;
}

function withTimeout<T>(promise: Promise<T>, ms: number) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error("Onboarding check timed out")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
