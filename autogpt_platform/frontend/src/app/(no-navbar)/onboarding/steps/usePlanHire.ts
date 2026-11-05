import {
  getBrainDumpPlan,
  getGetBrainDumpSuggestedTasksQueryKey,
  type hireBrainDumpPlanResponse,
  useGetBrainDumpPlan,
  useHireBrainDumpPlan,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { PlanHireResponse } from "@/app/api/__generated__/models/planHireResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useStableCallback } from "@/hooks/useStableCallback";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../store";
import { whenSkipSettled } from "./BrainDumpStep/useBrainDumpStep";
import {
  joinNames,
  pause,
  PLAN_POLL_MS,
  planAnsweredSince,
  readShownPlan,
} from "./SubscriptionStep/components/PlanPanel/helpers";

// The preparing screen holds its bar this long for the hire, then moves on
// whatever happened: a hire that never answers must not keep anyone from Home.
export const HIRE_MAX_WAIT_MS = 15_000;
// With no plan shown, how long the hire waits for the pipeline's: the hire
// settles the plan, so an early one loses the brain dump's. Short of
// HIRE_MAX_WAIT_MS, so the hire still goes out while the bar holds.
const PLAN_WAIT_MS = 10_000;
const HIRE_RETRY_MS = 2_000;
const HIRE_STARTED_KEY = "autogpt:onboarding-plan-hire:";

// Hires the plan's team once the trial (or the plan, without a paywall) is
// confirmed. Never before: a user who leaves Stripe hires nobody.
export function usePlanHire({ enabled }: { enabled: boolean }) {
  const userID = useOnboardingWizardStore((s) => s.userID);
  const queryClient = useQueryClient();
  const [mountedAt] = useState(() => Date.now());
  // A refresh mid-hire: the first call is still going on the server, so the
  // screen waits out what is left of its time before sending a second.
  const [resumedFor] = useState(() => hireTimeLeft(userID));
  const [settled, setSettled] = useState(false);
  const sent = useRef(false);
  const { mutateAsync: hirePlan } = useHireBrainDumpPlan();
  const plan = usePlanToHire({ enabled, since: mountedAt });
  const names = plan.team.map((expert) => expert.name);

  const hire = useStableCallback(async () => {
    try {
      const response = await hireUntilAnswered(
        () => sendMarked(userID, () => hirePlan()),
        mountedAt,
      );
      const failed = response.data.failed ?? [];
      if (failed.length > 0) toastHireFailed(failed.map((e) => e.name));
    } catch (error) {
      Sentry.captureException(error, { tags: { context: "plan-hire" } });
      toastHireFailed(names);
    } finally {
      // So Home's recipient picker and task owners see the new team, even
      // when a slow hire lands after Home has read them. Whatever the answer:
      // one lost on its way back may still have hired them.
      await Promise.all([
        invalidateExpertRosterQueries(queryClient),
        queryClient.invalidateQueries({
          queryKey: getGetBrainDumpSuggestedTasksQueryKey(),
        }),
      ]).catch(() => undefined);
      setSettled(true);
    }
  });

  // Its time up, ask how the first call went: the hire is idempotent, and
  // that answer is what refreshes Home and says who did not join.
  useEffect(() => {
    if (!enabled || resumedFor === null) return;
    const timer = setTimeout(() => void hire(), resumedFor);
    return () => clearTimeout(timer);
  }, [enabled, resumedFor, hire]);

  useEffect(() => {
    if (!enabled || resumedFor !== null || !plan.isKnown || sent.current) {
      return;
    }
    sent.current = true;
    void hire();
  }, [enabled, resumedFor, plan.isKnown, hire]);

  return {
    label: names.length > 0 ? `Hiring ${joinNames(names)}` : null,
    isSettled: !enabled || settled,
  };
}

// The plan whose team the server hires. One the plan step showed is it (only
// a stored plan is remembered); otherwise wait a while for the pipeline's.
function usePlanToHire({
  enabled,
  since,
}: {
  enabled: boolean;
  since: number;
}) {
  const userID = useOnboardingWizardStore((s) => s.userID);
  const [shown] = useState(() => readShownPlan(userID));
  const [isWaitOver, setIsWaitOver] = useState(false);
  const isWaiting = enabled && !shown;
  const query = useGetBrainDumpPlan({
    query: {
      enabled: isWaiting,
      // A skip still on its way would leave the server answering for the
      // take before it.
      queryFn: async ({ signal }) => {
        await whenSkipSettled();
        return getBrainDumpPlan({ signal });
      },
      refetchOnMount: "always",
      refetchInterval: (q) =>
        isWaitOver || planAnsweredSince(q.state, since)?.ready
          ? false
          : PLAN_POLL_MS,
    },
  });
  const answer = planAnsweredSince(query, since);
  const ready = shown ?? (answer?.ready ? answer : null);

  useEffect(() => {
    if (!isWaiting) return;
    const timer = setTimeout(() => setIsWaitOver(true), PLAN_WAIT_MS);
    return () => clearTimeout(timer);
  }, [isWaiting]);

  return { team: ready?.team ?? [], isKnown: Boolean(ready) || isWaitOver };
}

// The hire is idempotent, so one the server could not answer (a 5xx, no answer
// at all, or a plan still changing), or that nobody could join, is sent again
// while the preparing bar still waits for it.
async function hireUntilAnswered(
  hirePlan: () => Promise<hireBrainDumpPlanResponse>,
  since: number,
) {
  for (;;) {
    const isOutOfTime = () =>
      Date.now() - since + HIRE_RETRY_MS >= HIRE_MAX_WAIT_MS;
    try {
      const response = await hirePlan();
      if (response.status !== 200) throw new Error("Plan hire failed");
      if (!isEveryoneFailed(response.data) || isOutOfTime()) return response;
    } catch (error) {
      if (isRefused(error) || isOutOfTime()) throw error;
    }
    await pause(HIRE_RETRY_MS);
  }
}

function isEveryoneFailed({ hired, failed }: PlanHireResponse) {
  return (failed?.length ?? 0) > 0 && !hired?.length;
}

function isRefused(error: unknown) {
  return (
    error instanceof ApiError && error.status < 500 && error.status !== 409
  );
}

// Marked only while a try is on the wire: a refresh then waits for it
// instead of sending a second, and between tries sends the hire again.
async function sendMarked(
  userID: string | null,
  hirePlan: () => Promise<hireBrainDumpPlanResponse>,
) {
  markHireStarted(userID);
  try {
    return await hirePlan();
  } finally {
    clearHireStarted(userID);
  }
}

function toastHireFailed(names: string[]) {
  toast({
    title:
      names.length > 0
        ? `Couldn't hire ${joinNames(names)}`
        : "Couldn't hire your team",
    description:
      "Otto takes on their tasks for now. You can hire experts any time.",
    variant: "destructive",
    // Until closed: the screen moves on to Home, and this should be there.
    duration: Infinity,
  });
}

function hireTimeLeft(userID: string | null) {
  try {
    const at = Number(sessionStorage.getItem(`${HIRE_STARTED_KEY}${userID}`));
    const ago = Date.now() - at;
    return at > 0 && ago < HIRE_MAX_WAIT_MS ? HIRE_MAX_WAIT_MS - ago : null;
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
