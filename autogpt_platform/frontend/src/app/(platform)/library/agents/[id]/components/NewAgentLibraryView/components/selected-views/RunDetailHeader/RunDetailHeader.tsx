import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { ClockClockwiseIcon } from "@phosphor-icons/react";
import { formatDistanceToNow, formatDistanceStrict } from "date-fns";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../../helpers";
import { RunStatusBadge } from "../SelectedRunView/components/RunStatusBadge";

type Props = {
  agent: LibraryAgent;
  run: GraphExecution | undefined;
  scheduleRecurrence?: string;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
};

export function RunDetailHeader({ agent, run, scheduleRecurrence }: Props) {
  return (
    <div className={AGENT_LIBRARY_SECTION_PADDING_X}>
      <div className="flex w-full items-center justify-between">
        <div className="flex w-full flex-col gap-0">
          <div className="flex w-full flex-col flex-wrap items-start justify-between gap-1 md:flex-row md:items-center">
            <div className="flex min-w-0 flex-1 flex-col items-start gap-3">
              {run?.status ? (
                <RunStatusBadge status={run.status} />
              ) : scheduleRecurrence ? (
                <div className="inline-flex items-center gap-1 rounded-md bg-yellow-50 p-1">
                  <ClockClockwiseIcon
                    size={16}
                    className="text-yellow-700"
                    weight="bold"
                  />
                  <Text variant="small-medium" className="text-yellow-700">
                    Scheduled
                  </Text>
                </div>
              ) : null}
              <Text variant="h2" className="truncate text-ellipsis">
                {agent.name}
              </Text>
            </div>
          </div>
          {run ? (
            <div className="mt-1 flex flex-wrap items-center gap-2 gap-y-1 text-zinc-400">
              <Text variant="small" className="text-zinc-500">
                Started{" "}
                {run.started_at
                  ? formatDistanceToNow(run.started_at, { addSuffix: true })
                  : "—"}
              </Text>
              <span className="mx-1 inline-block text-zinc-200">|</span>
              <Text variant="small" className="text-zinc-500">
                Version: {run.graph_version}
              </Text>
              {run.stats?.node_exec_count !== undefined && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="text-zinc-500">
                    Steps: {run.stats.node_exec_count}
                  </Text>
                </>
              )}
              {run.stats?.duration !== undefined && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="text-zinc-500">
                    Duration:{" "}
                    {formatDistanceStrict(0, run.stats.duration * 1000)}
                  </Text>
                </>
              )}
              {run.stats?.cost !== undefined && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="text-zinc-500">
                    Cost: ${(run.stats.cost / 100).toFixed(2)}
                  </Text>
                </>
              )}
            </div>
          ) : scheduleRecurrence ? (
            <Text variant="small" className="mt-1 !text-zinc-600">
              {scheduleRecurrence}
            </Text>
          ) : null}
        </div>
      </div>
    </div>
  );
}
