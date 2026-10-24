import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
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

export function RunDetailHeader({ agent, run, scheduleRecurrence }: Props) {
  const isBillingEnabled = useIsBillingEnabled();
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
                    <div className="inline-flex items-center gap-1 rounded-md bg-warning-surface p-1">
                      <Icon
                        icon={FlaskConicalIcon}
                        size={16}
                        className="text-warning-foreground"
                      />
                      <Text
                        variant="small-medium"
                        className="!text-warning-foreground"
                      >
                        Simulated
                      </Text>
                    </div>
                  )}
                </div>
              ) : scheduleRecurrence ? (
                <div
                  // TODO(tokens): no exact role for bg-yellow-50 (#FEF9E6)
                  className="inline-flex items-center gap-1 rounded-md bg-yellow-50 p-1"
                >
                  <Icon
                    icon={TimeScheduleIcon}
                    size={16}
                    // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
                    className="text-yellow-700"
                  />
                  <Text
                    // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
                    variant="small-medium"
                    className="text-yellow-700"
                  >
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
            <div className="mt-1 flex flex-wrap items-center gap-2 gap-y-1 text-foreground-subtle">
              <Text variant="small" className="text-foreground-tertiary">
                Started{" "}
                {run.started_at
                  ? formatDistanceToNow(run.started_at, { addSuffix: true })
                  : "—"}
              </Text>
              <span
                // TODO(tokens): no fitting role for text-zinc-200 #DADADC; #DADADC is surface-active, a surface fill, and border-base, a border colour
                className="mx-1 inline-block text-zinc-200"
              >
                |
              </span>
              <Text variant="small" className="text-foreground-tertiary">
                Version: {run.graph_version}
              </Text>
              {run.stats?.node_exec_count !== undefined && (
                <>
                  <span
                    // TODO(tokens): no fitting role for text-zinc-200 #DADADC; #DADADC is surface-active, a surface fill, and border-base, a border colour
                    className="mx-1 inline-block text-zinc-200"
                  >
                    |
                  </span>
                  <Text variant="small" className="text-foreground-tertiary">
                    Steps: {run.stats.node_exec_count}
                  </Text>
                </>
              )}
              {run.stats?.duration !== undefined && (
                <>
                  <span
                    // TODO(tokens): no fitting role for text-zinc-200 #DADADC; #DADADC is surface-active, a surface fill, and border-base, a border colour
                    className="mx-1 inline-block text-zinc-200"
                  >
                    |
                  </span>
                  <Text variant="small" className="text-foreground-tertiary">
                    Duration:{" "}
                    {formatDistanceStrict(0, run.stats.duration * 1000)}
                  </Text>
                </>
              )}
              {isBillingEnabled && run.stats?.cost !== undefined && (
                <>
                  <span
                    // TODO(tokens): no fitting role for text-zinc-200 #DADADC; #DADADC is surface-active, a surface fill, and border-base, a border colour
                    className="mx-1 inline-block text-zinc-200"
                  >
                    |
                  </span>
                  <Text variant="small" className="text-foreground-tertiary">
                    Cost: ${(run.stats.cost / 100).toFixed(2)}
                  </Text>
                </>
              )}
            </div>
          ) : scheduleRecurrence ? (
            <Text variant="small" className="mt-1 !text-foreground-secondary">
              {scheduleRecurrence}
            </Text>
          ) : null}
        </div>
      </div>
    </div>
  );
}
