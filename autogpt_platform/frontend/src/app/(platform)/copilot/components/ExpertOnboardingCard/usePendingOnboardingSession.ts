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
  // A refetch, and a failed one, keep serving the cached answer, which can name
  // a chat whose card was settled since.
  const isSettled =
    (query.isFetchedAfterMount || query.isError) && !query.isFetching;
  const sessionId =
    isEnabled && isSettled && query.isSuccess && query.data?.status === 200
      ? (query.data.data?.session_id ?? null)
      : null;
  return { sessionId, isChecking: isEnabled && !isSettled };
}
