"use client";

import { useListExpertRuns } from "@/app/api/__generated__/endpoints/experts/experts";
import { type ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { okData } from "@/app/api/helpers";
import { isOutputType } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { useState } from "react";
import { FilterIconMenu } from "../FilterIconMenu";
import {
  filterExpertRuns,
  getWorkEmptyMessage,
  WORK_FILTERS,
  WorkFilter,
  getRunMeta,
} from "./helpers";

interface Props {
  expertId: string;
  expertName: string;
  enabled: boolean;
}

export function ExpertWorkSection({ expertId, expertName, enabled }: Props) {
  const [filter, setFilter] = useState<WorkFilter>("all");
  const [activeRun, setActiveRun] = useState<ExpertRun | null>(null);

  const runsQuery = useListExpertRuns(expertId, {
    query: { select: (res) => okData(res) ?? null, enabled },
  });
  const runs = runsQuery.data ?? [];
  const visibleRuns = filterExpertRuns(runs, filter);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="body-medium" tone="primary">
          {expertName}&apos;s Work
        </Text>
        <FilterIconMenu
          label="Filter work"
          value={filter}
          defaultValue="all"
          options={WORK_FILTERS}
          onChange={setFilter}
        />
      </div>

      {runsQuery.isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-16 w-full rounded-lg" />
          <Skeleton className="h-16 w-full rounded-lg" />
        </div>
      ) : runsQuery.isError && runsQuery.data == null ? (
        <ErrorCard
          context="this expert's work"
          hint="We could not load this expert's recent work."
          onRetry={() => runsQuery.refetch()}
        />
      ) : visibleRuns.length === 0 ? (
        <Text variant="body" tone="muted">
          {getWorkEmptyMessage(filter)}
        </Text>
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
  const meta = getRunMeta(run);
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg bg-white p-3 ring-1 ring-inset ring-zinc-200">
      <div className="min-w-0">
        <Text variant="body-medium" tone="primary" className="truncate">
          {run.agent_name}
        </Text>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <RunStatusBadge status={run.status} />
          {run.needs_review && run.status.toUpperCase() !== "REVIEW" ? (
            <Badge variant="warning" size="small">
              Needs review
            </Badge>
          ) : null}
          {meta.parts.length > 0 ? (
            <Text
              variant="small"
              as="span"
              tone="muted"
              title={meta.startedAt?.toLocaleString()}
              data-testid="expert-run-meta"
            >
              {meta.parts.join(" · ")}
            </Text>
          ) : null}
        </div>
      </div>
      <Button variant="secondary" size="xs" onClick={onOpen}>
        Open
      </Button>
    </div>
  );
}
