"use client";

import { useListExpertRuns } from "@/app/api/__generated__/endpoints/experts/experts";
import { type ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { okData } from "@/app/api/helpers";
import { isOutputType } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { Button } from "@/components/atoms/Button/Button";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { cn } from "@/lib/utils";
import { useState } from "react";

interface Props {
  expertId: string;
  enabled: boolean;
}

export function ExpertWorkSection({ expertId, enabled }: Props) {
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false);
  const [activeRun, setActiveRun] = useState<ExpertRun | null>(null);

  const runsQuery = useListExpertRuns(expertId, {
    query: { select: (res) => okData(res) ?? null, enabled },
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
      ) : runsQuery.isError && runsQuery.data == null ? (
        <ErrorCard
          context="this expert's work"
          hint="We could not load this expert's recent work."
          onRetry={() => runsQuery.refetch()}
        />
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
          outputKey={activeRun.output_key}
          graphId={activeRun.graph_id}
          executionId={activeRun.execution_id}
          runLink={activeRun.link}
        />
      ) : null}
    </section>
  );
}

function ExpertRunRow({ run, onOpen }: { run: ExpertRun; onOpen: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-white p-3 ring-1 ring-inset ring-zinc-200">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-zinc-900">
          {run.agent_name}
        </p>
        <div className="mt-1 flex items-center gap-2">
          <RunStatusBadge status={run.status} />
          {run.needs_review && run.status.toUpperCase() !== "REVIEW" ? (
            <Badge variant="warning" size="small">
              Needs review
            </Badge>
          ) : null}
        </div>
      </div>
      <Button variant="secondary" size="small" onClick={onOpen}>
        Open
      </Button>
    </div>
  );
}
