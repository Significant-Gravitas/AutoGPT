import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertRun } from "@/app/api/__generated__/models/expertRun";

export interface ExpertTask {
  run: ExpertRun;
  expert: Expert;
}

/** Newest first. Runs with no start time sort last rather than jumping to the
 *  top on a NaN comparison. */
export function sortTasksByRecency(tasks: ExpertTask[]) {
  return [...tasks].sort((a, b) => startedAtMs(b.run) - startedAtMs(a.run));
}

function startedAtMs(run: ExpertRun) {
  if (!run.started_at) return Number.NEGATIVE_INFINITY;
  const time = new Date(run.started_at).valueOf();
  return Number.isNaN(time) ? Number.NEGATIVE_INFINITY : time;
}
