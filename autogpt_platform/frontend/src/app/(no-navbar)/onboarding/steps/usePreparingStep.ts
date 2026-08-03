import { useGetBrainDumpRecommendedProviders } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { peekIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  "Connecting automation engines",
  "Building your space",
] as const;

// Honest copy for the dump path — these steps are things that are
// actually happening on the server while the bar fills. The last one
// gates on the provider-recommendation job so the connect dialog never
// opens on an empty "Recommended" section.
const BRAIN_DUMP_CHECKLIST = [
  "Reading your brain dump",
  "Briefing AutoPilot on your work",
  "Building your space",
  "Finding tools for your work",
] as const;

// The generic path keeps its original fixed duration; the dump path runs
// longer because its steps mirror real server work.
const GENERIC_DURATION_MS = 4_000;
const BRAIN_DUMP_DURATION_MS = 10_000;

const RECOMMENDATIONS_POLL_MS = 2_500;
// A job the backend never finished (process restart mid-run) must not
// strand the user on this screen — advance anyway after this ceiling.
const RECOMMENDATIONS_MAX_WAIT_MS = 60_000;
// While waiting on the job the bar parks just short of full so it still
// reads as "almost there", not "stuck at done".
const WAITING_PROGRESS_CAP = 95;

export function usePreparingStep({
  onComplete,
  isBrainDumpEnabled,
}: {
  onComplete: () => void;
  isBrainDumpEnabled: boolean;
}) {
  const [started, setStarted] = useState(false);
  const [completedItems, setCompletedItems] = useState(0);
  const [progress, setProgress] = useState(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  // Only a user who actually dumped gets the honest copy; a skip would
  // make "Reading your brain dump" a lie.
  const isDumpPath = isBrainDumpEnabled && peekIntroPath() === "A";
  const checklist = isDumpPath ? BRAIN_DUMP_CHECKLIST : GENERIC_CHECKLIST;
  const duration = isDumpPath ? BRAIN_DUMP_DURATION_MS : GENERIC_DURATION_MS;
  const stepInterval = duration / checklist.length;

  const recommendedQuery = useGetBrainDumpRecommendedProviders({
    query: {
      enabled: isDumpPath,
      refetchInterval: (query) => {
        const response = query.state.data;
        if (response && (response.status !== 200 || response.data.ready)) {
          return false;
        }
        return RECOMMENDATIONS_POLL_MS;
      },
    },
  });
  // A non-200 answer counts as ready: the endpoint won't recover inside
  // this screen, and the dialog copes with an empty recommendation list.
  const isRecommendationsReady =
    !isDumpPath ||
    (recommendedQuery.data !== undefined &&
      (recommendedQuery.data.status !== 200 ||
        recommendedQuery.data.data.ready));
  const isRecommendationsReadyRef = useRef(isRecommendationsReady);
  isRecommendationsReadyRef.current = isRecommendationsReady;

  useEffect(() => {
    const timer = setTimeout(() => setStarted(true), 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!started) return;

    const startTime = Date.now();
    let finished = false;

    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const isWaitingOnRecommendations =
        !isRecommendationsReadyRef.current &&
        elapsed < RECOMMENDATIONS_MAX_WAIT_MS;

      if (elapsed >= duration && !isWaitingOnRecommendations && !finished) {
        finished = true;
        clearInterval(progressInterval);
        setProgress(100);
        setCompletedItems(checklist.length);
        onCompleteRef.current();
        return;
      }

      const pct = Math.min(100, (elapsed / duration) * 100);
      const items = Math.min(
        checklist.length,
        Math.floor(elapsed / stepInterval) + 1,
      );
      // The last step stays unchecked while its work is genuinely still
      // running server-side.
      setProgress(
        isWaitingOnRecommendations ? Math.min(pct, WAITING_PROGRESS_CAP) : pct,
      );
      setCompletedItems(
        isWaitingOnRecommendations
          ? Math.min(items, checklist.length - 1)
          : items,
      );
    }, 50);

    return () => clearInterval(progressInterval);
  }, [started, checklist.length, stepInterval, duration]);

  return { started, progress, completedItems, checklist };
}
