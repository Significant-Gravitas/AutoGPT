"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { useNow } from "@/hooks/useNow";
import { formatDistanceToNow, formatDistanceStrict } from "date-fns";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../../helpers";
import { RunStatusBadge } from "../SelectedRunView/components/RunStatusBadge";
import { FlaskConicalIcon, TimeScheduleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { isCompletedWithErrors } from "@/lib/run-health";

type Props = {
  agent: LibraryAgent;
  run: GraphExecution | undefined;
  scheduleRecurrence?: string;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
};

function formatRunDuration(run: GraphExecution, nowMs: number): string | null {
  const isLive =
    run.status === AgentExecutionStatus.RUNNING ||
    run.status === AgentExecutionStatus.QUEUED;

  // Finished: prefer finalized stats.duration when present.
  if (!isLive && run.stats?.duration != null) {
    return formatDistanceStrict(0, run.stats.duration * 1000);
  }

  // Live (or finished without stats): started_at → ended_at|now.
  if (!run.started_at) return null;
  const start = new Date(run.started_at).getTime();
  if (Number.isNaN(start)) return null;
  const end = run.ended_at ? new Date(run.ended_at).getTime() : nowMs;
  if (Number.isNaN(end) || end < start) return null;
  return formatDistanceStrict(0, end - start);
}

export function RunDetailHeader({ agent, run, scheduleRecurrence }: Props) {
  const isBillingEnabled = useIsBillingEnabled();
  const isLive =
    run?.status === AgentExecutionStatus.RUNNING ||
    run?.status === AgentExecutionStatus.QUEUED;
  const nowMs = useNow(1000, Boolean(isLive && run?.started_at));
  const durationLabel = run ? formatRunDuration(run, nowMs) : null;

  return (
    <div className={AGENT_LIBRARY_SECTION_PADDING_X}>
      <div className="flex w-full items-center justify-between">
        <div className="flex w-full flex-col gap-0">
          <div className="flex w-full flex-col flex-wrap items-start justify-between gap-1 md:flex-row md:items-center">
            <div className="flex min-w-0 flex-1 flex-col items-start gap-3">
              {run?.status ? (
                <div className="flex items-center gap-2">
                  <RunStatusBadge
                    status={run.status}
                    completedWithErrors={isCompletedWithErrors(run)}
                  />
                  {run.is_dry_run && (
                    <div className="inline-flex items-center gap-1 rounded-md bg-warning-surface p-1 text-warning-foreground">
                      <Icon icon={FlaskConicalIcon} size={16} />
                      <Text variant="small-medium" tone="warning">
                        Simulated
                      </Text>
                    </div>
                  )}
                </div>
              ) : scheduleRecurrence ? (
                <div
                  // oxlint-disable-next-line design-system/no-raw-colors -- yellow-700 #AE8800 has no role; mustard to orange-brown change deferred (see PR)
                  className="inline-flex items-center gap-1 rounded-md bg-warning-surface p-1 text-yellow-700"
                >
                  <Icon icon={TimeScheduleIcon} size={16} />
                  <Text variant="small-medium" tone="inherit">
                    Scheduled
                  </Text>
                </div>
              ) : null}
              <Text variant="h2">{agent.name}</Text>
            </div>
          </div>
          {run ? (
            <div className="mt-1 flex flex-wrap items-center gap-2 gap-y-1 text-foreground-subtle">
              <Text variant="small" tone="muted">
                Started{" "}
                {run.started_at
                  ? formatDistanceToNow(run.started_at, { addSuffix: true })
                  : "—"}
              </Text>
              <span className="mx-1 inline-block text-border-base">|</span>
              <Text variant="small" tone="muted">
                Version: {run.graph_version}
              </Text>
              {run.stats?.node_exec_count !== undefined && (
                <>
                  <span className="mx-1 inline-block text-border-base">|</span>
                  <Text variant="small" tone="muted">
                    Steps: {run.stats.node_exec_count}
                  </Text>
                </>
              )}
              {durationLabel && (
                <>
                  <span className="mx-1 inline-block text-border-base">|</span>
                  <Text variant="small" tone="muted">
                    Duration: {durationLabel}
                  </Text>
                </>
              )}
              {isBillingEnabled && run.stats?.cost !== undefined && (
                <>
                  <span className="mx-1 inline-block text-border-base">|</span>
                  <Text variant="small" tone="muted">
                    Cost: ${(run.stats.cost / 100).toFixed(2)}
                  </Text>
                </>
              )}
            </div>
          ) : scheduleRecurrence ? (
            <Text variant="small" tone="secondary" className="mt-1">
              {scheduleRecurrence}
            </Text>
          ) : null}
        </div>
      </div>
    </div>
  );
}
