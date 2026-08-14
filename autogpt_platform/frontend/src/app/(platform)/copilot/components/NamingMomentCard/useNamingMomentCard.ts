import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  isNamingMomentEligible,
  peekNamingMomentDismissed,
  setNamingMomentDismissed,
} from "./helpers";

export function useNamingMomentCard() {
  const router = useRouter();
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const isExpertsEnabled = Boolean(enabled);

  const [dismissedUserId, setDismissedUserId] = useState<string | null>(() =>
    peekNamingMomentDismissed(userId) ? userId : null,
  );
  const isDismissed = Boolean(
    userId && (dismissedUserId === userId || peekNamingMomentDismissed(userId)),
  );

  // Dismissal is known synchronously from localStorage, so a permanently
  // dismissed user never pays for the experts/sessions probes on every
  // empty-state mount just to compute an eligibility that is already false.
  const queriesEnabled =
    Boolean(userId) && isExpertsEnabled && ready && !isDismissed;

  const expertsQuery = useListExperts({
    query: {
      enabled: queriesEnabled,
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });

  const sessionsQuery = useGetV2ListSessions(
    { limit: 1 },
    { query: { enabled: queriesEnabled, refetchOnWindowFocus: false } },
  );

  const hasExperts = (expertsQuery.data?.length ?? 0) > 0;
  const sessionsTotal =
    sessionsQuery.data?.status === 200 ? sessionsQuery.data.data.total : 0;
  const isLoaded = expertsQuery.isSuccess && sessionsQuery.isSuccess;

  const isEligible = isNamingMomentEligible({
    isExpertsEnabled,
    isFlagReady: ready,
    isLoaded,
    hasExperts,
    hasSessions: sessionsTotal > 0,
    isDismissed,
  });

  function dismiss() {
    setNamingMomentDismissed(userId);
    setDismissedUserId(userId);
  }

  function startNaming() {
    router.push("/raise?from=naming");
  }

  return { isEligible, dismiss, startNaming };
}
