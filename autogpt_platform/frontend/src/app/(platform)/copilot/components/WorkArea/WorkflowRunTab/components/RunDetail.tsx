"use client";

import {
  getGetV1GetExecutionDetailsQueryKey,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowLeft01Icon, StopCircleIcon } from "@hugeicons/core-free-icons";
import { useQueryClient } from "@tanstack/react-query";
import {
  InfoLoading,
  InfoMeta,
} from "../../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockNote } from "../../blocks";
import { formatCost, formatWhen, runElapsedMs, triggerLabel } from "../runs";
import { formatDuration, type Trace } from "../trace";
import { countLabel } from "../value";
import { StatusPill } from "./StatusPill";
import { ValueSection } from "./StepDetail";
import { TraceWaterfall } from "./TraceWaterfall";

interface Props {
  graphID: string;
  title: string;
  runID: string;
  run: GraphExecution | undefined;
  trace: Trace;
  isLive: boolean;
  now: number;
  onBack: () => void;
}

/** One run on its own page of the tab: which run it is and how it ended,
 *  what it was given, the trace of its blocks and what it gave back. */
export function RunDetail({
  graphID,
  title,
  runID,
  run,
  trace,
  isLive,
  now,
  onBack,
}: Props) {
  const queryClient = useQueryClient();
  const stop = usePostV1StopGraphExecution({
    mutation: {
      onSuccess: (_, { graphExecId }) =>
        queryClient.invalidateQueries({
          queryKey: getGetV1GetExecutionDetailsQueryKey(graphID, graphExecId),
        }),
    },
  });

  const back = (
    <Button
      variant="ghost-muted"
      size="xs"
      leadingIcon={ArrowLeft01Icon}
      className="-ml-1.5 self-start"
      onClick={onBack}
    >
      All runs
    </Button>
  );

  if (!run) {
    return (
      <div className="flex flex-col gap-3">
        <div className="flex px-1">{back}</div>
        <InfoLoading />
      </div>
    );
  }

  const blockCount = trace.steps.length;
  const failedCount = trace.steps.filter(
    (step) => step.state === "failed",
  ).length;
  const cost = formatCost(run.stats?.cost);
  const elapsed = runElapsedMs(run, now);

  return (
    <div aria-label={`Run of ${title}`} className="flex flex-col gap-4 px-1">
      <div className="flex flex-col gap-1">
        {back}
        <Text variant="body-md-medium" as="h2" tone="ink" className="truncate">
          {title}
        </Text>
        <Text variant="small" as="p" tone="quiet">
          {triggerLabel(run)} · {formatWhen(run.started_at)}
        </Text>
      </div>

      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <StatusPill status={run.status} />
        {failedCount > 0 ? (
          <Text variant="small-medium" as="span" tone="error">
            {countLabel(failedCount, "block failed", "blocks failed")}
          </Text>
        ) : null}
        {elapsed !== null ? (
          <Text variant="small" as="span" tone="text" tabular>
            {formatDuration(elapsed)}
          </Text>
        ) : null}
        {cost ? (
          <Text variant="small" as="span" tone="quiet" tabular>
            {cost}
          </Text>
        ) : null}
        {isLive ? (
          <Button
            variant="ghost-muted"
            size="xs"
            leadingIcon={StopCircleIcon}
            loading={stop.isPending}
            onClick={() =>
              stop.mutate({ graphId: graphID, graphExecId: runID })
            }
          >
            Stop
          </Button>
        ) : null}
      </div>

      <ValueSection label="Input" value={run.inputs ?? null} />

      <section aria-label="Trace" className="flex flex-col gap-1">
        <div className="flex h-6 items-center justify-between">
          <Text variant="small-medium" as="h4" tone="quiet">
            Trace
          </Text>
          {blockCount > 0 ? (
            <InfoMeta>
              {countLabel(blockCount, "block run", "block runs")}
            </InfoMeta>
          ) : null}
        </div>
        {blockCount === 0 ? (
          <BlockNote>
            <Text variant="inbox" tone="quiet">
              {isLive
                ? "Waiting for the first block to start…"
                : "This run has no block runs to show."}
            </Text>
          </BlockNote>
        ) : (
          <TraceWaterfall key={runID} trace={trace} />
        )}
      </section>

      {isLive ? null : (
        <ValueSection label="Output" value={run.outputs ?? null} />
      )}
    </div>
  );
}
