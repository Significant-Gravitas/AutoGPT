import {
  getBrainDumpPlan,
  type getBrainDumpPlanResponse,
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
  PLAN_MAX_WAIT_MS,
  PLAN_POLL_MS,
  readShownPlan,
  rememberShownPlan,
} from "./helpers";

// The plan step's plan: polled while the pipeline writes it, shown once it is
// ready or once the wait gives up, and never swapped after that. Giving up
// settles the plan on the server, so the trial and Home use the one shown.
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
  const mountedAt = useRef(0);
  const reported = useRef(false);

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
        givenUp || readyAnswer(q.state.data) ? false : PLAN_POLL_MS,
    },
  });
  const answer = query.isFetchedAfterMount ? answerOf(query.data) : null;
  const plan = restored ?? givenUp ?? (answer?.ready ? answer : null);
  const isPending = !plan;

  useEffect(() => {
    mountedAt.current = Date.now();
  }, []);

  useEffect(() => {
    latest.current = answer;
  }, [answer]);

  const giveUp = useStableCallback(async () => {
    const settled = await settleBrainDumpPlan().catch(() => null);
    // The plan landed while the settle was on its way, and is already shown.
    if (latest.current?.ready) return;
    setGivenUp(
      settled?.status === 200 ? settled.data : (latest.current ?? EMPTY_PLAN),
    );
  });

  useEffect(() => {
    if (!isPending) return;
    const timer = setTimeout(() => void giveUp(), PLAN_MAX_WAIT_MS);
    return () => clearTimeout(timer);
  }, [isPending, giveUp]);

  useEffect(() => {
    if (!plan || restored || reported.current) return;
    reported.current = true;
    rememberShownPlan(userID, plan);
    trackPlan("plan_viewed", {
      plan_source: plan.source,
      skipped_dump: plan.skipped_dump,
      tasks: getStepTasks(plan).length,
      experts: plan.team?.length ?? 0,
      wait_ms: Math.max(0, Date.now() - mountedAt.current),
      gave_up: givenUp !== null,
    });
  }, [plan, restored, givenUp, userID]);

  return { plan, isPending, progress: plan ? null : answer?.progress };
}

function answerOf(response: getBrainDumpPlanResponse | undefined) {
  return response?.status === 200 ? response.data : null;
}

function readyAnswer(response: getBrainDumpPlanResponse | undefined) {
  return Boolean(answerOf(response)?.ready);
}
