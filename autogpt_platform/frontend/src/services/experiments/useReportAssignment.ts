"use client";

import { usePostExperimentsRecordExperimentAssignment } from "@/app/api/__generated__/endpoints/experiments/experiments";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useEffect } from "react";

export type AssignmentSource = "posthog" | "launchdarkly";

interface Args {
  experimentKey: string;
  variant: string | null;
  isResolved: boolean;
  source: AssignmentSource;
}

/**
 * Report an experiment arm to the backend once per user and experiment, so
 * the assignment is available to the `analytics.*` views whichever tool did
 * the bucketing. Only string arms are reported: an unresolved or
 * not-enrolled flag is not an assignment.
 */
export function useReportAssignment({
  experimentKey,
  variant,
  isResolved,
  source,
}: Args) {
  const { user } = useAuth();
  const { mutateAsync: recordAssignment } =
    usePostExperimentsRecordExperimentAssignment({
      mutation: { retry: false },
    });
  const userID = user?.id ?? null;

  useEffect(() => {
    if (!isResolved || !variant || !userID) return;
    const claim = claimAssignmentReport(userID, experimentKey);
    if (!claim) return;
    const reportUserID = userID;
    const reportClaim = claim;
    const data = { experiment_key: experimentKey, variant, source };
    let active = true;
    let succeeded = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    async function report() {
      retryTimer = undefined;
      try {
        await recordAssignment({ data });
        succeeded = true;
      } catch (error) {
        failures += 1;
        if (!active || failures > 2 || !isRetryable(error)) {
          releaseAssignmentReport(reportUserID, experimentKey, reportClaim);
          return;
        }
        retryTimer = setTimeout(report, 1000 * 2 ** (failures - 1));
      }
    }

    retryTimer = setTimeout(report, 0);
    return () => {
      active = false;
      if (retryTimer !== undefined) {
        clearTimeout(retryTimer);
      }
      if (!succeeded) {
        releaseAssignmentReport(userID, experimentKey, claim);
      }
    };
  }, [isResolved, variant, userID, experimentKey, source, recordAssignment]);
}

function isRetryable(error: unknown) {
  return (
    !(error instanceof ApiError) ||
    error.status === 408 ||
    error.status === 429 ||
    error.status >= 500
  );
}

const reportedAssignments = new Map<string, symbol>();

function claimAssignmentReport(userID: string, experimentKey: string) {
  const key = `${userID}:${experimentKey}`;
  if (reportedAssignments.has(key)) return;
  const claim = Symbol();
  reportedAssignments.set(key, claim);
  return claim;
}

function releaseAssignmentReport(
  userID: string,
  experimentKey: string,
  claim: symbol,
) {
  const key = `${userID}:${experimentKey}`;
  if (reportedAssignments.get(key) === claim) reportedAssignments.delete(key);
}

export function resetReportedAssignmentsForTests() {
  reportedAssignments.clear();
}
