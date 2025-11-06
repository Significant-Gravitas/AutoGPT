import React from "react";
import { RunStatusBadge } from "../SelectedRunView/components/RunStatusBadge";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import {
  TrashIcon,
  StopIcon,
  PlayIcon,
  ArrowSquareOutIcon,
} from "@phosphor-icons/react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import moment from "moment";
import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { useRunDetailHeader } from "./useRunDetailHeader";
import { AgentActionsDropdown } from "../AgentActionsDropdown";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ShareRunButton } from "../ShareRunButton/ShareRunButton";

type Props = {
  agent: LibraryAgent;
  run: GraphExecution | undefined;
  scheduleRecurrence?: string;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
};

export function RunDetailHeader({
  agent,
  run,
  scheduleRecurrence,
  onSelectRun,
  onClearSelectedRun,
}: Props) {
  const shareExecutionResultsEnabled = useGetFlag(Flag.SHARE_EXECUTION_RESULTS);

  const {
    canStop,
    isStopping,
    isDeleting,
    isRunning,
    isRunningAgain,
    openInBuilderHref,
    showDeleteDialog,
    handleStopRun,
    handleRunAgain,
    handleDeleteRun,
    handleShowDeleteDialog,
  } = useRunDetailHeader(agent.graph_id, run, onSelectRun, onClearSelectedRun);

  return (
    <div>
      <div className="flex w-full items-center justify-between">
        <div className="flex w-full flex-col gap-0">
          <div className="flex w-full flex-col flex-wrap items-start justify-between gap-2 md:flex-row md:items-center">
            <div className="flex min-w-0 flex-1 flex-col items-start gap-2 md:flex-row md:items-center">
              {run?.status ? <RunStatusBadge status={run.status} /> : null}
              <Text
                variant="h3"
                className="truncate text-ellipsis !font-normal"
              >
                {agent.name}
              </Text>
            </div>
            {run ? (
              <div className="my-4 flex flex-wrap items-center gap-2 md:my-2 lg:my-0">
                <Button
                  variant="secondary"
                  size="small"
                  onClick={handleRunAgain}
                  loading={isRunningAgain}
                >
                  <PlayIcon size={16} /> Run again
                </Button>
                {shareExecutionResultsEnabled && (
                  <ShareRunButton
                    graphId={agent.graph_id}
                    executionId={run.id}
                    isShared={run.is_shared}
                    shareToken={run.share_token}
                  />
                )}
                {!isRunning ? (
                  <Button
                    variant="secondary"
                    size="small"
                    onClick={() => handleShowDeleteDialog(true)}
                  >
                    <TrashIcon size={16} /> Delete run
                  </Button>
                ) : null}
                {openInBuilderHref ? (
                  <Button
                    variant="secondary"
                    size="small"
                    as="NextLink"
                    href={openInBuilderHref}
                    target="_blank"
                  >
                    <ArrowSquareOutIcon size={16} /> Edit run
                  </Button>
                ) : null}
                {canStop ? (
                  <Button
                    variant="destructive"
                    size="small"
                    onClick={handleStopRun}
                    disabled={isStopping}
                  >
                    <StopIcon size={14} /> Stop agent
                  </Button>
                ) : null}
                <AgentActionsDropdown agent={agent} />
              </div>
            ) : null}
          </div>
          {run ? (
            <div className="mt-1 flex flex-wrap items-center gap-2 gap-y-1 text-zinc-600">
              <Text variant="small" className="!text-zinc-600">
                Started {moment(run.started_at).fromNow()}
              </Text>
              <span className="mx-1 inline-block text-zinc-200">|</span>
              <Text variant="small" className="!text-zinc-600">
                Version: {run.graph_version}
              </Text>
              {run.stats?.node_exec_count !== undefined && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="!text-zinc-600">
                    Steps: {run.stats.node_exec_count}
                  </Text>
                </>
              )}
              {run.stats?.duration !== undefined && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="!text-zinc-600">
                    Duration:{" "}
                    {moment.duration(run.stats.duration, "seconds").humanize()}
                  </Text>
                </>
              )}
              {run.stats?.cost !== undefined && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="!text-zinc-600">
                    Cost: ${(run.stats.cost / 100).toFixed(2)}
                  </Text>
                </>
              )}
              {run.stats?.activity_status && (
                <>
                  <span className="mx-1 inline-block text-zinc-200">|</span>
                  <Text variant="small" className="!text-zinc-600">
                    {String(run.stats.activity_status)}
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

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: handleShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete run"
      >
        <Dialog.Content>
          <div>
            <Text variant="large">
              Are you sure you want to delete this run? This action cannot be
              undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeleting}
                onClick={() => handleShowDeleteDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteRun}
                loading={isDeleting}
              >
                Delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
