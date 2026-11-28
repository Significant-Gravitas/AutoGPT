import type { RunsSkippedProblem } from "@/app/api/__generated__/models/runsSkippedProblem";

// A schedule's key and inputs can't be changed here yet, so the fix offered
// is a new schedule, as in the alert email.
export const SKIPPED_RUNS_COPY: Record<
  RunsSkippedProblem,
  { reason: string; fix: string }
> = {
  key_removed: {
    reason: "The key it used was removed.",
    fix: "Schedule the workflow again with another key, then delete this schedule.",
  },
  key_unusable: {
    reason: "It can’t use one of its keys.",
    fix: "Schedule the workflow again with a key that works, then delete this schedule.",
  },
  inputs_invalid: {
    reason: "Its inputs no longer pass the workflow’s checks.",
    fix: "Schedule the workflow again with inputs that do, then delete this schedule.",
  },
  step_disabled: {
    reason: "One of its steps is switched off.",
    fix: "It runs again once the step is back on. To run it before then, replace the step and schedule the workflow again.",
  },
};
