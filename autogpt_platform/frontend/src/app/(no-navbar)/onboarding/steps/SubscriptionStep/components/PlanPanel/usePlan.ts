import {
  getBrainDumpPlan,
  settleBrainDumpPlan,
  useGetBrainDumpPlan,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import { useStableCallback } from "@/hooks/useStableCallback";
import { trackPlan } from "@/services/onboarding/plan-analytics";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../../../store";
import { whenSkipSettled } from "../../../BrainDumpStep/useBrainDumpStep";
import {
  EMPTY_PLAN,
  getStepTasks,
  isCheckoutReturn,
  pause,
  PLAN_MAX_WAIT_MS,
  PLAN_POLL_MS,
  planAnsweredSince,
  readShownPlan,
  rememberShownPlan,
} from "./helpers";

const SETTLE_ATTEMPTS = 3;
const SETTLE_TIMEOUT_MS = 5_000;
const SKIP_WAIT_MS = 5_000;

// The plan step's plan: polled while the pipeline writes it, shown once it is
// ready or once the wait gives up. Giving up settles the plan on the server,
// so the trial and Home use the one shown, and a settled plan is never
// swapped. One the server never stored is, by its plan once that is ready.
export function usePlan() {
  const userID = useOnboardingWizardStore((s) => s.userID);
  const searchParams = useSearchParams();
  // Back from Stripe without paying: the plan the user just saw, at once,
  // with no second view reported.
  const [restored] = useState(() =>
    isCheckoutReturn(searchParams) ? readShownPlan(userID) : null,
  );
  const [givenUp, setGivenUp] = useState<PlanResponse | null>(null);
  const latest = useRef<PlanResponse | null>(null);
  const [mountedAt] = useState(() => Date.now());
  const reported = useRef(false);
  const waitRanOut = useRef(false);

  const query = useGetBrainDumpPlan({
    query: {
      enabled: !restored,
      // A skip still on its way would leave the server answering for the
      // take the user went Back from.
      queryFn: async ({ signal }) => {
        await whenSkipSettled();
        return getBrainDumpPlan({ signal });
      },
      // Back to the brain dump and a re-record changes the plan, so a plan
      // cached before this mount is never what the step shows.
      refetchOnMount: "always",
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      refetchInterval: (q) =>
        givenUp?.ready || planAnsweredSince(q.state, mountedAt)?.ready
          ? false
          : PLAN_POLL_MS,
    },
  });
  const answer = planAnsweredSince(query, mountedAt);
  const ready = answer?.ready ? answer : null;
  const plan = restored ?? (givenUp?.ready ? givenUp : (ready ?? givenUp));
  const isPending = !plan;

  useEffect(() => {
    latest.current = answer;
  }, [answer]);

  const giveUp = useStableCallback(async () => {
    waitRanOut.current = true;
    // Until a skip lands, the server holds the take the user went Back from,
    // which a settle would store; the polls then carry on behind a fallback.
    const settled = (await skipLandsWithin(SKIP_WAIT_MS))
      ? await settlePlan(() => Boolean(latest.current?.ready))
      : null;
    // The plan landed while the settle was on its way, and is already shown.
    if (latest.current?.ready) return;
    setGivenUp(settled ?? latest.current ?? EMPTY_PLAN);
  });

  useEffect(() => {
    if (!isPending) return;
    const timer = setTimeout(() => void giveUp(), PLAN_MAX_WAIT_MS);
    return () => clearTimeout(timer);
  }, [isPending, giveUp]);

  useEffect(() => {
    if (plan && !restored) rememberShownPlan(userID, plan);
  }, [plan, restored, userID]);

  useEffect(() => {
    if (!plan || restored || reported.current) return;
    reported.current = true;
    trackPlan("plan_viewed", {
      plan_source: plan.source,
      skipped_dump: plan.skipped_dump,
      tasks: getStepTasks(plan).length,
      experts: plan.team?.length ?? 0,
      wait_ms: Math.max(0, Date.now() - mountedAt),
      gave_up: waitRanOut.current,
    });
  }, [plan, restored, mountedAt]);

  return { plan, isPending, progress: plan ? null : answer?.progress };
}

// The trial hires what the settle stores, so a failed one is tried again,
// and each is bounded: one that never answers must not keep the step waiting.
async function settlePlan(isPlanShown: () => boolean) {
  let settled: PlanResponse | null = null;
  for (let attempt = 0; attempt < SETTLE_ATTEMPTS; attempt++) {
    if (attempt > 0) await pause(PLAN_POLL_MS);
    if (isPlanShown()) return null;
    settled = (await settleOnce()) ?? settled;
    if (settled?.ready) return settled;
  }
  return settled;
}

function skipLandsWithin(ms: number) {
  return Promise.race([
    whenSkipSettled().then(() => true),
    pause(ms).then(() => false),
  ]);
}

async function settleOnce() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SETTLE_TIMEOUT_MS);
  try {
    const response = await settleBrainDumpPlan({ signal: controller.signal });
    return response.status === 200 ? response.data : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
