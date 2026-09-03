"use client";

import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import type { SitrepItemData } from "@/app/(platform)/library/types";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { RunChip } from "./components/RunChip";

interface Props {
  runs: SitrepItemData[];
}

/** The last few days of workflow activity, as a strip under the briefing's
 *  outcomes. Runs come from the library's own execution feed rather than the
 *  home aggregate, so a workflow that never produced an outcome still shows
 *  its latest state here. */
export function RecentWorkflowRuns({ runs }: Props) {
  if (runs.length === 0) return null;

  return (
    <section
      aria-label="Recent workflow runs"
      className="flex flex-col gap-2 border-t border-zinc-100 pt-4"
    >
      <div className="flex items-center justify-between gap-3">
        <Text variant="small" className="font-medium text-zinc-500">
          Recent workflow runs
        </Text>
        <NextLink
          href="/library"
          className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-700"
        >
          View all <Icon icon={ArrowRight02Icon} size={12} />
        </NextLink>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300">
        {runs.map((run) => (
          <RunChip key={run.id} run={run} />
        ))}
      </div>
    </section>
  );
}
