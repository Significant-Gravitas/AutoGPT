"use client";

import { useListExpertRuns } from "@/app/api/__generated__/endpoints/experts/experts";
import { type ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { WorkOutputSheet } from "@/app/(platform)/copilot/components/WorkOutputSheet/WorkOutputSheet";
import { isOutputType } from "@/app/(platform)/copilot/components/WorkOutputSheet/helpers";

interface Props {
  expertId: string;
  enabled: boolean;
}

export function ExpertWorkSection({ expertId, enabled }: Props) {
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false);
  const [activeRun, setActiveRun] = useState<ExpertRun | null>(null);

  const runsQuery = useListExpertRuns(expertId, {
    query: { select: (res) => okData(res) ?? [], enabled },
  });
  const runs = runsQuery.data ?? [];
  const reviewCount = runs.filter((run) => run.needs_review).length;
  const visibleRuns = needsReviewOnly
    ? runs.filter((run) => run.needs_review)
    : runs;

  return (
    <section>
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
          Work
        </div>
        {reviewCount > 0 ? (
          <button
            type="button"
            aria-pressed={needsReviewOnly}
            onClick={() => setNeedsReviewOnly((value) => !value)}
            className={cn(
              "rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset transition-colors",
              needsReviewOnly
                ? "bg-amber-100 text-amber-700 ring-amber-200"
                : "bg-white text-zinc-500 ring-zinc-200 hover:text-zinc-800",
            )}
          >
            Needs review ({reviewCount})
          </button>
        ) : null}
      </div>

      {runsQuery.isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-16 w-full rounded-xl" />
          <Skeleton className="h-16 w-full rounded-xl" />
        </div>
      ) : visibleRuns.length === 0 ? (
        <p className="text-sm text-zinc-500">
          {needsReviewOnly
            ? "Nothing is waiting on your review."
            : "No completed work yet. Finished runs will show up here."}
        </p>
      ) : (
        <ul className="flex flex-col gap-3" aria-label="Expert work">
          {visibleRuns.map((run) => (
            <li key={run.execution_id}>
              <ExpertRunRow run={run} onOpen={() => setActiveRun(run)} />
            </li>
          ))}
        </ul>
      )}

      {activeRun ? (
        <WorkOutputSheet
          open={activeRun !== null}
          onOpenChange={(open) => {
            if (!open) setActiveRun(null);
          }}
          title={activeRun.agent_name}
          outputType={
            isOutputType(activeRun.output_type)
              ? activeRun.output_type
              : "unknown"
          }
          graphId={activeRun.graph_id}
          executionId={activeRun.execution_id}
          runLink={activeRun.link}
        />
      ) : null}
    </section>
  );
}

function ExpertRunRow({ run, onOpen }: { run: ExpertRun; onOpen: () => void }) {
  const failed = run.status.toUpperCase().includes("FAILED");
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-white p-3 ring-1 ring-inset ring-zinc-200">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-zinc-900">
          {run.agent_name}
        </p>
        <div className="mt-1 flex items-center gap-2">
          <span
            className={cn(
              "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
              failed
                ? "bg-red-50 text-red-600"
                : "bg-emerald-50 text-emerald-600",
            )}
          >
            {failed ? "Failed" : "Completed"}
          </span>
          {run.needs_review ? (
            <span className="inline-flex items-center rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700">
              Needs review
            </span>
          ) : null}
        </div>
      </div>
      <Button variant="secondary" size="small" onClick={onOpen}>
        Open
      </Button>
    </div>
  );
}
