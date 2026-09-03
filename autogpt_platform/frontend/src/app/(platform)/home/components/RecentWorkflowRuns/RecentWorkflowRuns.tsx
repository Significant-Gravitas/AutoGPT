"use client";

import type { SitrepItemData } from "@/app/(platform)/library/types";
import { Text } from "@/components/atoms/Text/Text";
import { RunRow } from "./components/RunRow";

interface Props {
  runs: SitrepItemData[];
}

/** The last few days of workflow activity, as rows under the briefing's
 *  outcomes. Runs come from the library's own execution feed rather than the
 *  home aggregate, so a workflow that never produced an outcome still shows
 *  its latest state here. */
export function RecentWorkflowRuns({ runs }: Props) {
  if (runs.length === 0) return null;

  return (
    <section
      aria-label="Recent workflow runs"
      className="border-t border-zinc-100 pb-1"
    >
      <Text
        variant="small-medium"
        className="px-4 pb-1 pt-3 text-[11px] uppercase tracking-[0.06em] text-zinc-400"
      >
        Recent workflow runs
      </Text>
      <div>
        {runs.map((run) => (
          <RunRow key={run.id} run={run} />
        ))}
      </div>
    </section>
  );
}
