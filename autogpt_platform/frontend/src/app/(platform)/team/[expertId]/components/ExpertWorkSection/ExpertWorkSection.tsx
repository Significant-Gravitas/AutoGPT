"use client";

import {
  useListExpertRuns,
  useListExpertWork,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { type ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { type ExpertWorkItem } from "@/app/api/__generated__/models/expertWorkItem";
import { okData } from "@/app/api/helpers";
import { isOutputType } from "@/components/organisms/WorkOutputSheet/helpers";
import { WorkOutputSheet } from "@/components/organisms/WorkOutputSheet/WorkOutputSheet";
import { Button } from "@/components/atoms/Button/Button";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { cn } from "@/lib/utils";
import {
  founderSafeArtifactName,
  founderSafeText,
} from "@/lib/founder-safe-text";
import { parseWorkspaceURI } from "@/lib/workspace-uri";
import { getGetWorkspaceDownloadFileByIdUrl } from "@/app/api/__generated__/endpoints/workspace/workspace";
import Link from "next/link";
import { useEffect, useState } from "react";

interface Props {
  expertId: string;
  enabled: boolean;
}

const EMPTY_WORK_ITEMS: ExpertWorkItem[] = [];

export function ExpertWorkSection({ expertId, enabled }: Props) {
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false);
  const [activeRun, setActiveRun] = useState<ExpertRun | null>(null);

  const runsQuery = useListExpertRuns(expertId, {
    query: {
      select: (res) => okData(res) ?? null,
      enabled,
      refetchInterval: 5_000,
    },
  });
  const workQuery = useListExpertWork(expertId, {
    query: {
      select: (res) => okData(res) ?? null,
      enabled,
      refetchInterval: 5_000,
    },
  });
  const runs = runsQuery.data ?? [];
  const workItems = workQuery.data ?? EMPTY_WORK_ITEMS;
  const reviewCount = runs.filter((run) => run.needs_review).length;
  const visibleRuns = needsReviewOnly
    ? runs.filter((run) => run.needs_review)
    : runs;

  function handleRetry() {
    runsQuery.refetch();
    workQuery.refetch();
  }

  useEffect(() => {
    if (workItems.length === 0 || typeof window === "undefined") return;
    const target = new URLSearchParams(window.location.search).get(
      "workItemId",
    );
    if (!target) return;
    document
      .getElementById(`work-item-${target}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [workItems]);

  return (
    <section>
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
          Work history
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

      {runsQuery.isLoading || workQuery.isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-16 w-full rounded-xl" />
          <Skeleton className="h-16 w-full rounded-xl" />
        </div>
      ) : (runsQuery.isError && runsQuery.data == null) ||
        (workQuery.isError && workQuery.data == null) ? (
        <ErrorCard
          context="this expert's work"
          hint="We could not load this expert's recent work."
          onRetry={handleRetry}
        />
      ) : visibleRuns.length === 0 && workItems.length === 0 ? (
        <p className="text-sm text-zinc-500">
          {needsReviewOnly
            ? "Nothing is waiting on your review."
            : "No work yet. Delegated tasks and workflow runs will show up here."}
        </p>
      ) : (
        <ul className="flex flex-col gap-3" aria-label="Expert work">
          {!needsReviewOnly
            ? workItems.map((work) => (
                <li key={work.id} id={`work-item-${work.id}`}>
                  <ExpertWorkRow work={work} />
                </li>
              ))
            : null}
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

function ExpertWorkRow({ work }: { work: ExpertWorkItem }) {
  return (
    <article className="rounded-xl bg-white p-3 ring-1 ring-inset ring-zinc-200 target:ring-2 target:ring-zinc-900">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-zinc-900">
            {founderSafeText(work.task_title, "Expert work")}
          </p>
          <p className="mt-1 line-clamp-2 text-sm text-zinc-500">
            {founderSafeText(
              work.result || work.blocker || work.expected_deliverable,
              "AutoPilot is reviewing this work.",
            )}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Badge variant={workVariant(work.status)} size="small">
              {workStatusLabel(work.status)}
            </Badge>
            <span className="text-xs text-zinc-400">
              {confidenceLabel(work.confidence)}
            </span>
            {work.artifacts.length > 0 ? (
              <span className="text-xs text-zinc-400">
                {work.artifacts.length}{" "}
                {work.artifacts.length === 1 ? "deliverable" : "deliverables"}
              </span>
            ) : null}
          </div>
          {work.artifacts.length > 0 ? (
            <ul
              className="mt-2 flex flex-wrap gap-1.5"
              aria-label="Deliverables"
            >
              {work.artifacts.map((artifact) => (
                <ArtifactItem
                  key={`${artifact.name}-${artifact.uri}`}
                  name={artifact.name}
                  uri={artifact.uri}
                />
              ))}
            </ul>
          ) : null}
        </div>
        <Link
          href={`/copilot?sessionId=${encodeURIComponent(work.delegated_session_id)}`}
          className="shrink-0 rounded-lg px-2.5 py-1.5 text-sm font-medium text-zinc-700 ring-1 ring-inset ring-zinc-200 transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
        >
          Open thread
        </Link>
      </div>
    </article>
  );
}

function ArtifactItem({ name, uri }: { name: string; uri: string }) {
  const workspace = parseWorkspaceURI(uri);
  const label = founderSafeArtifactName(name);
  const className =
    "block max-w-full truncate rounded-md bg-zinc-50 px-2 py-1 text-xs text-zinc-600";
  return (
    <li className="max-w-full">
      {workspace ? (
        <a
          href={getGetWorkspaceDownloadFileByIdUrl(workspace.fileID)}
          className={`${className} transition-colors hover:bg-zinc-100 hover:text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400`}
        >
          {label}
        </a>
      ) : (
        <span className={className}>{label}</span>
      )}
    </li>
  );
}

function workStatusLabel(status: ExpertWorkItem["status"]) {
  if (status === "blocked_manager") return "Needs AutoPilot";
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function confidenceLabel(confidence: ExpertWorkItem["confidence"]) {
  return confidence.charAt(0).toUpperCase() + confidence.slice(1);
}

function workVariant(status: ExpertWorkItem["status"]) {
  if (status === "delivered") return "success" as const;
  if (status === "failed") return "error" as const;
  if (status === "partial" || status === "blocked_manager") {
    return "warning" as const;
  }
  return "info" as const;
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
