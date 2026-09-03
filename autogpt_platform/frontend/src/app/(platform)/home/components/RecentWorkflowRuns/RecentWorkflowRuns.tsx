"use client";

import NextLink from "next/link";
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
      <div className="flex items-center justify-between px-4 pb-1 pt-3">
        <Text
          variant="small-medium"
          className="text-[11px] uppercase tracking-[0.06em] text-zinc-400"
        >
          Recent workflow runs
        </Text>
        <NextLink
          href="/library"
          className="text-[11px] font-medium text-zinc-500 outline-none transition-colors hover:text-zinc-900 focus-visible:underline"
        >
          View all
        </NextLink>
      </div>
      <div>
        {runs.map((run) => (
          <RunRow key={run.id} run={run} />
        ))}
      </div>
    </section>
  );
}
