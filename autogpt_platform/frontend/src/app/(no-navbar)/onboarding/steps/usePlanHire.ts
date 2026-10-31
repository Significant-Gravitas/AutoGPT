import {
  getGetBrainDumpPlanQueryOptions,
  useGetBrainDumpPlan,
  useHireBrainDumpPlan,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useStableCallback } from "@/hooks/useStableCallback";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { trackPlan } from "@/services/onboarding/plan-analytics";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../store";
import {
  joinNames,
  readShownPlan,
} from "./SubscriptionStep/components/PlanPanel/helpers";

// The preparing screen holds its bar this long for the hire, then moves on
// whatever happened: a hire that never answers must not keep anyone from Home.
export const HIRE_MAX_WAIT_MS = 15_000;
const HIRE_STARTED_KEY = "autogpt:onboarding-plan-hire:";

// Hires the plan's team once the trial (or the plan, without a paywall) is
// confirmed. Never before: a user who leaves Stripe hires nobody.
export function usePlanHire({ enabled }: { enabled: boolean }) {
  const userID = useOnboardingWizardStore((s) => s.userID);
  const queryClient = useQueryClient();
  const [settled, setSettled] = useState(false);
  const started = useRef(false);
  const { mutateAsync: hirePlan } = useHireBrainDumpPlan();
  // The server hires its current plan's team. A ready plan the step showed
  // is that plan; one it gave up on may since have been replaced, so ask.
  const [shown] = useState(() => {
    const plan = readShownPlan(userID);
    return plan?.ready ? plan : null;
  });
  const { data } = useGetBrainDumpPlan({
    query: { enabled: enabled && !shown, refetchOnMount: "always" },
  });
  const team = shown?.team ?? (data?.status === 200 ? data.data.team : []);
  const names = (team ?? []).map((expert) => expert.name);

  const hire = useStableCallback(async () => {
    try {
      const response = await hirePlan();
      if (response.status !== 200) throw new Error("Plan hire failed");
      trackPlan("plan_experts_hired", {
        hired: response.data.hired?.length ?? 0,
        failed: response.data.failed?.length ?? 0,
        via: "preparing",
      });
      // So Home's recipient picker and task owners see the new team.
      await invalidateExpertRosterQueries(queryClient).catch(() => undefined);
    } catch (error) {
      Sentry.captureException(error, { tags: { context: "plan-hire" } });
      trackPlan("plan_experts_hired", {
        hired: 0,
        failed: await teamSize(),
        via: "preparing",
      });
    } finally {
      clearHireStarted(userID);
      setSettled(true);
    }
  });

  useEffect(() => {
    if (!enabled || started.current) return;
    started.current = true;
    // A refresh mid-hire: the first call is still going on the server, so
    // wait out what is left of its time instead of sending a second one.
    const since = hireStartedAgo(userID);
    if (since !== null && since < HIRE_MAX_WAIT_MS) {
      const timer = setTimeout(
        () => setSettled(true),
        HIRE_MAX_WAIT_MS - since,
      );
      return () => clearTimeout(timer);
    }
    markHireStarted(userID);
    void hire();
  }, [enabled, userID, hire]);

  // Everyone failed to join; how many that is may still be loading.
  async function teamSize() {
    if (shown) return shown.team?.length ?? 0;
    const plan = await queryClient
      .fetchQuery(getGetBrainDumpPlanQueryOptions())
      .catch(() => null);
    return plan?.status === 200 ? (plan.data.team?.length ?? 0) : 0;
  }

  return {
    label: names.length > 0 ? `Hiring ${joinNames(names)}` : null,
    isSettled: !enabled || settled,
  };
}

function hireStartedAgo(userID: string | null) {
  try {
    const at = Number(sessionStorage.getItem(`${HIRE_STARTED_KEY}${userID}`));
    return at > 0 ? Date.now() - at : null;
  } catch {
    return null;
  }
}

function markHireStarted(userID: string | null) {
  try {
    sessionStorage.setItem(`${HIRE_STARTED_KEY}${userID}`, String(Date.now()));
  } catch {
    // Without sessionStorage a refresh may send the hire again; it is
    // idempotent, so that costs a request, not a duplicate expert.
  }
}

function clearHireStarted(userID: string | null) {
  try {
    sessionStorage.removeItem(`${HIRE_STARTED_KEY}${userID}`);
  } catch {
    // See markHireStarted.
  }
}
