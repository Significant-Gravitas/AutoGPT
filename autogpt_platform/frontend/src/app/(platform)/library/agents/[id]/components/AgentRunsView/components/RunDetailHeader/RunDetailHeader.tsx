import { RunStatusBadge } from "../RunDetails/components/RunStatusBadge";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import {
  PencilSimpleIcon,
  TrashIcon,
  StopIcon,
  PlayIcon,
  ArrowSquareOut,
} from "@phosphor-icons/react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import moment from "moment";
import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { useRunDetailHeader } from "./useRunDetailHeader";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import Link from "next/link";

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
  const {
    stopRun,
    canStop,
    isStopping,
    deleteRun,
    isDeleting,
    runAgain,
    isRunningAgain,
    openInBuilderHref,
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
                  onClick={runAgain}
                  loading={isRunningAgain}
                >
                  <PlayIcon size={16} /> Run again
                </Button>
                <Button
                  variant="secondary"
                  size="small"
                  onClick={deleteRun}
                  loading={isDeleting}
                >
                  <TrashIcon size={16} /> Delete run
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="secondary" size="small">
                      •••
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    {canStop ? (
                      <DropdownMenuItem onClick={stopRun} disabled={isStopping}>
                        <StopIcon size={14} className="mr-2" /> Stop run
                      </DropdownMenuItem>
                    ) : null}
                    {openInBuilderHref ? (
                      <DropdownMenuItem asChild>
                        <Link
                          href={openInBuilderHref}
                          target="_blank"
                          className="flex items-center gap-2"
                        >
                          <ArrowSquareOut size={14} /> Open in builder
                        </Link>
                      </DropdownMenuItem>
                    ) : null}
                    <DropdownMenuItem asChild>
                      <Link
                        href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
                        target="_blank"
                        className="flex items-center gap-2"
                      >
                        <PencilSimpleIcon size={16} /> Edit agent
                      </Link>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
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
                    Cost: ${run.stats.cost}
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
    </div>
  );
}
