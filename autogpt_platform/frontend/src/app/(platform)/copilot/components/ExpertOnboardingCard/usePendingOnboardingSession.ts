import { useGetExpertOnboarding } from "@/app/api/__generated__/endpoints/experts/experts";

/** The chat whose onboarding card still waits on the user, so an empty chat
 *  with that expert can open it instead of starting a second one. */
export function usePendingOnboardingSession(
  expertId: string | null,
  enabled: boolean,
) {
  const isEnabled = enabled && !!expertId;
  const query = useGetExpertOnboarding(expertId ?? "", {
    query: {
      enabled: isEnabled,
      staleTime: 0,
      refetchOnMount: "always",
    },
  });
  const isSettled = query.isFetchedAfterMount || query.isError;
  const sessionId =
    isEnabled && query.isFetchedAfterMount && query.data?.status === 200
      ? (query.data.data?.session_id ?? null)
      : null;
  return { sessionId, isChecking: isEnabled && !isSettled };
}
