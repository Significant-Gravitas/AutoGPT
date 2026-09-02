"use client";

import { usePostV2RecordExperimentAssignment } from "@/app/api/__generated__/endpoints/experiments/experiments";
import { useAuth } from "@/lib/auth/hooks/useAuth";
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
  const { mutate: recordAssignment } = usePostV2RecordExperimentAssignment();
  const userID = user?.id ?? null;

  useEffect(() => {
    if (!isResolved || !variant || !userID) return;
    if (!claimAssignmentReport(userID, experimentKey)) return;
    recordAssignment({
      data: { experiment_key: experimentKey, variant, source },
    });
  }, [isResolved, variant, userID, experimentKey, source, recordAssignment]);
}

const reportedAssignments = new Set<string>();

function claimAssignmentReport(userID: string, experimentKey: string) {
  const key = `${userID}:${experimentKey}`;
  if (reportedAssignments.has(key)) return false;
  reportedAssignments.add(key);
  return true;
}

export function resetReportedAssignmentsForTests() {
  reportedAssignments.clear();
}
