import { getV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

// Each full list reads at most this many of the runs it leaves out.
const MAX_CHECKS_PER_LIST = 10;

interface Args {
  runs: GraphExecutionMeta[];
  log: {
    startRead(runID: string): void;
    finishRead(runID: string, read?: GraphExecutionMeta): boolean;
  };
  onRead: (run: GraphExecutionMeta) => void;
  onGone: (runID: string) => void;
}

/**
 * A full run list holds only the newest runs, so it can't tell whether an
 * older run shown as going has ended. One that ended while live updates were
 * lost would show as going for good, so the full list read after a reconnect
 * reads such runs by ID, the ones read longest ago (or never) first. A live
 * update or a list that comes after a read started is newer than it and
 * stands.
 *
 * Each read is a request of its own: through the query cache it could join
 * one already on its way (the run's view polls the same query), whose answer
 * may predate news this read must not overrule.
 */
export function createOmittedRunChecker() {
  const reading = new Set<string>();
  // run id -> when it was last read, as a count of reads
  const lastRead = new Map<string, number>();
  let reads = 0;

  return {
    async check({ runs, log, onRead, onGone }: Args) {
      const leftOut = new Set(runs.map((run) => run.id));
      for (const id of Array.from(lastRead.keys())) {
        if (!leftOut.has(id)) lastRead.delete(id);
      }
      const due = runs
        .filter((run) => !reading.has(run.id))
        .sort((a, b) => (lastRead.get(a.id) ?? 0) - (lastRead.get(b.id) ?? 0))
        .slice(0, MAX_CHECKS_PER_LIST);
      await Promise.all(
        due.map(async (run) => {
          reading.add(run.id);
          lastRead.set(run.id, ++reads);
          log.startRead(run.id);
          let read: GraphExecutionMeta | undefined;
          let gone = false;
          try {
            const details = okData(
              await getV1GetExecutionDetails(run.graph_id, run.id),
            );
            read = details && metaOf(details);
          } catch (error) {
            gone = error instanceof ApiError && error.status === 404;
            // Any other failure keeps the run as shown; a later list reads it.
          } finally {
            reading.delete(run.id);
          }
          if (!log.finishRead(run.id, read)) return;
          if (read) onRead(read);
          else if (gone) onGone(run.id);
        }),
      );
    },
  };
}

// The details endpoint is the only read of a single run. Activity keeps its
// meta fields, not the run's outputs and node executions.
function metaOf(
  details: GraphExecution | GraphExecutionWithNodes,
): GraphExecutionMeta {
  const meta: GraphExecutionMeta &
    Partial<Pick<GraphExecutionWithNodes, "outputs" | "node_executions">> = {
    ...details,
  };
  delete meta.outputs;
  delete meta.node_executions;
  return meta;
}
