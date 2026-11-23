import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import { okData } from "@/app/api/helpers";
import { createWallClock, useWallClock } from "@/hooks/useNowSecond";
import { keepPreviousData } from "@tanstack/react-query";
import { buildTrace } from "./trace";

/** The running bar's clock: four ticks a second, shared by every open trace. */
const traceClock = createWallClock(250);

interface Args {
  graphID: string;
  graphVersion: number | null;
  run: GraphExecution | GraphExecutionWithNodes | undefined;
  isLive: boolean;
}

/** A run's trace: its block runs named and coloured from the block catalogue,
 *  with the workflow's other blocks waiting while it is live, on a clock that
 *  ticks only then. */
export function useRunTrace({ graphID, graphVersion, run, isLive }: Args) {
  const executions = run && "node_executions" in run ? run.node_executions : [];
  const nodes =
    useGetV1GetSpecificGraph(
      graphID,
      { version: graphVersion },
      {
        query: {
          enabled: !!run,
          select: okData,
          staleTime: Infinity,
          retry: false,
        },
      },
    ).data?.nodes ?? [];

  const blockIDs = Array.from(
    new Set([
      ...nodes.map((node) => node.block_id),
      ...executions.map((execution) => execution.block_id),
    ]),
  ).sort();
  const blocks =
    useGetV2GetSpecificBlocks(
      { block_ids: blockIDs },
      {
        query: {
          enabled: blockIDs.length > 0,
          select: okData,
          staleTime: Infinity,
          placeholderData: keepPreviousData,
        },
      },
    ).data ?? [];

  const now = useWallClock(traceClock, isLive);
  const trace = buildTrace({
    executions,
    nodes,
    blocks,
    runStartedAt: run?.started_at,
    runEndedAt: run?.ended_at,
    isLive,
    now,
  });
  return { trace, now };
}
