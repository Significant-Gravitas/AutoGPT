import {
  getBrainDumpPlan,
  getGetBrainDumpSuggestedTasksQueryKey,
  hireBrainDumpPlan,
  useGetBrainDumpPlan,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useStableCallback } from "@/hooks/useStableCallback";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../store";
import { whenSkipSettled } from "./BrainDumpStep/useBrainDumpStep";
import {
  hireTimeLeft,
  hireUntilAnswered,
  sendMarked,
  toastHireFailed,
} from "./plan-hire-helpers";
import {
  joinNames,
  PLAN_POLL_MS,
  planAnsweredSince,
  readShownPlan,
} from "./SubscriptionStep/components/PlanPanel/helpers";

// With no plan shown, how long the hire waits for the pipeline's: the hire
// settles the plan, so an early one loses the brain dump's. Short of
// HIRE_MAX_WAIT_MS, so the hire still goes out while the bar holds.
const PLAN_WAIT_MS = 10_000;

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
  const plan = usePlanToHire({ enabled, since: mountedAt });
  const names = plan.team.map((expert) => expert.name);

  const hire = useStableCallback(async () => {
    try {
      const response = await hireUntilAnswered(
        (signal) => sendMarked(userID, () => hireBrainDumpPlan({ signal })),
        // A re-ask goes out with the bar's time nearly spent, and may be the
        // only try that does the hire: it keeps trying from when it starts.
        resumedFor === null ? mountedAt : Date.now(),
      );
      const failed = response.data.failed ?? [];
      if (failed.length > 0) toastHireFailed(failed.map((e) => e.name));
    } catch (error) {
      Sentry.captureException(error, { tags: { context: "plan-hire" } });
      // After a refresh the first call may well have hired them: failing to
      // ask says nothing about how it went.
      if (resumedFor === null) toastHireFailed(names);
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

  // Its time up, ask how the first call went: the hire is idempotent, so this
  // does the hire if that call was lost, and its answer is what refreshes Home
  // and says who did not join.
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
