import {
  getGetOnboardingInterviewDetailsQueryKey,
  useGetOnboardingInterviewDetails,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { UIDataTypes, UIMessage, UITools } from "ai";

// The arms whose screens show what the interview heard.
const DETAILS_ARMS = new Set([
  "02-profile-card",
  "03-summary-chips",
  "06-coverage-ring",
  "08-transcript-highlights",
]);

interface Args {
  sessionId: string | null;
  arm: string | null;
  turns: number;
  isTurnRunning: boolean;
}

// What the typed interview has heard, read once per answer of the person's
// for an arm that shows it; the server keeps each read.
export function useInterviewDetails({
  sessionId,
  arm,
  turns,
  isTurnRunning,
}: Args) {
  const query = useGetOnboardingInterviewDetails({
    query: {
      queryKey: [...getGetOnboardingInterviewDetailsQueryKey(), turns],
      enabled:
        !!sessionId &&
        !!arm &&
        DETAILS_ARMS.has(arm) &&
        turns > 0 &&
        !isTurnRunning,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      placeholderData: (previous) => previous,
    },
  });
  return {
    data: query.data?.status === 200 ? (query.data.data.details ?? null) : null,
  };
}

export function answeredTurns(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  return messages.filter((message) => message.role === "user").length;
}
