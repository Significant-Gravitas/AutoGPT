import { useGetBrainDumpRecommendedProviders } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { peekIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";
import { HIRE_MAX_WAIT_MS, usePlanHire } from "./usePlanHire";

// The second item is the plan's team being hired (HIRE_ITEM), named once
// the plan is known.
const HIRE_ITEM = 1;
const TEAM_FALLBACK = "Setting up your team";

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  TEAM_FALLBACK,
  "Building your space",
] as const;

// Honest copy for the dump path — these steps are things that are
// actually happening on the server while the bar fills. The last one
// gates on the provider-recommendation job so the connect dialog never
// opens on an empty "Recommended" section.
const BRAIN_DUMP_CHECKLIST = [
  "Reading your brain dump",
  TEAM_FALLBACK,
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
// The recommender runs on the fast model, so a job still unanswered this
// late is one that is not coming; the connect dialog falls back to popular
// providers rather than making the user watch a bar for another minute.
const RECOMMENDATIONS_MAX_WAIT_MS = 15_000;
// While waiting on the job the bar parks just short of full so it still
// reads as "almost there", not "stuck at done".
const WAITING_PROGRESS_CAP = 95;

interface Args {
  onComplete: () => void;
  /** True once the trial or plan is confirmed, or with no paywall at all:
   *  the only moment the plan's team may be hired. */
  canHire?: boolean;
}

export function usePreparingStep({ onComplete, canHire = false }: Args) {
  const [started, setStarted] = useState(false);
  const [completedItems, setCompletedItems] = useState(0);
  const [progress, setProgress] = useState(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const teamHire = usePlanHire({ enabled: canHire });
  const isHireSettledRef = useRef(teamHire.isSettled);
  useEffect(() => {
    isHireSettledRef.current = teamHire.isSettled;
  }, [teamHire.isSettled]);

  // Only a user who actually dumped gets the honest copy; a skip would
  // make "Reading your brain dump" a lie.
  const isDumpPath = peekIntroPath() === "A";
  const checklist: readonly string[] = (
    isDumpPath ? BRAIN_DUMP_CHECKLIST : GENERIC_CHECKLIST
  ).map((item, index) =>
    index === HIRE_ITEM ? (teamHire.label ?? TEAM_FALLBACK) : item,
  );
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
      const isWaitingOnHire =
        !isHireSettledRef.current && elapsed < HIRE_MAX_WAIT_MS;
      const isWaiting = isWaitingOnRecommendations || isWaitingOnHire;

      if (elapsed >= duration && !isWaiting && !finished) {
        finished = true;
        clearInterval(progressInterval);
        setProgress(100);
        setCompletedItems(checklist.length);
        onCompleteRef.current();
        return;
      }

      const pct = Math.min(100, (elapsed / duration) * 100);
      const items = Math.min(
        checklist.length - 1,
        Math.floor(elapsed / stepInterval),
        // The hire item keeps its spinner until the hire answers.
        isWaitingOnHire ? HIRE_ITEM : checklist.length,
      );
      // The last step stays unchecked while its work is genuinely still
      // running server-side.
      setProgress(isWaiting ? Math.min(pct, WAITING_PROGRESS_CAP) : pct);
      setCompletedItems(items);
    }, 50);

    return () => clearInterval(progressInterval);
  }, [started, checklist.length, stepInterval, duration]);

  return { started, progress, completedItems, checklist };
}
