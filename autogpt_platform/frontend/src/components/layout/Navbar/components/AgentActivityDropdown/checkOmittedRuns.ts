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
    /** Starts a read by ID and returns its registration token. */
    startRead(runID: string): number;
    /**
     * Settles the read that `token` started. Rejects a completion whose
     * registration was replaced or cleared.
     */
    finishRead(
      runID: string,
      token: number,
      read?: GraphExecutionMeta,
      gone?: boolean,
    ): { accepted: boolean; supersededAt?: number };
    currentSequence(): number;
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
  // Whether a reconnect check has run: only then does a later refresh
  // carry runs it never got to. Before any reconnect, live updates keep
  // the shown runs current, so no read is worth taking.
  let checked = false;
  // Runs a newer check asked to read again once the read now out settles,
  // with the log's sequence when the check asked: a newer list or live
  // update that settles the run first drops the ask, but only when that
  // observation is newer than the ask — an earlier one predates the
  // reconnect the ask belongs to, so the run may have ended since, and the
  // fresh read still has to run. The ask takes no sequence number of its
  // own, so an observation with the same number arrived before it.
  const queued = new Map<string, number>();
  // Runs whose last read failed (a 500, say), with how many times each has
  // been retried. A success, or a 404, clears the count.
  const failures = new Map<string, number>();
  const MAX_FAILED_ATTEMPTS = 3;

  async function readOne(
    run: GraphExecutionMeta,
    { log, onRead, onGone }: Pick<Args, "log" | "onRead" | "onGone">,
  ): Promise<void> {
    reading.add(run.id);
    lastRead.set(run.id, ++reads);
    const token = log.startRead(run.id);
    let read: GraphExecutionMeta | undefined;
    let gone = false;
    try {
      const details = okData(
        await getV1GetExecutionDetails(run.graph_id, run.id),
      );
      read = details && metaOf(details);
      failures.delete(run.id);
    } catch (error) {
      gone = error instanceof ApiError && error.status === 404;
      // Any other failure keeps the run as shown; a later list reads it.
      if (!gone) {
        failures.set(run.id, (failures.get(run.id) ?? 0) + 1);
      } else {
        failures.delete(run.id);
      }
    } finally {
      reading.delete(run.id);
    }
    const { accepted, supersededAt } = log.finishRead(
      run.id,
      token,
      read,
      gone,
    );
    if (accepted) {
      if (read) onRead(read);
      else if (gone) onGone(run.id);
    }
    // This check's answer predates a newer check that asked for the run
    // again, so it can't satisfy that reconnect: read it fresh. A run
    // that is gone is not read again. A newer list or live update that
    // settled the run first drops the ask — but only when the observation
    // that settled it is newer than the ask: an earlier observation
    // predates the reconnect the ask belongs to, so the fresh read still
    // has to run. An observation whose time is unknown drops the ask, as
    // before.
    const queuedAt = queued.get(run.id);
    if (queuedAt === undefined) return;
    queued.delete(run.id);
    if (!accepted && (supersededAt === undefined || supersededAt > queuedAt)) {
      return;
    }
    if (!gone) {
      await readOne(run, { log, onRead, onGone });
    }
  }

  return {
    async check({ runs, log, onRead, onGone }: Args) {
      checked = true;
      const leftOut = new Set(runs.map((run) => run.id));
      for (const id of Array.from(lastRead.keys())) {
        if (!leftOut.has(id)) lastRead.delete(id);
      }
      const due = runs
        .filter((run) => !reading.has(run.id))
        .sort((a, b) => (lastRead.get(a.id) ?? 0) - (lastRead.get(b.id) ?? 0))
        .slice(0, MAX_CHECKS_PER_LIST);
      // A run already being read gets a fresh read once that one settles:
      // its answer predates this check, so it can't satisfy this reconnect.
      for (const run of runs) {
        if (reading.has(run.id)) queued.set(run.id, log.currentSequence());
      }
      await Promise.all(
        due.map((run) => readOne(run, { log, onRead, onGone })),
      );
    },

    /**
     * The follow-up to a later list refresh: only the runs the reconnect
     * check never got to (past the first ten) and the reads that failed,
     * for those runs only, a bounded number of times. Runs already read
     * are not read again here.
     */
    async followUp({ runs, log, onRead, onGone }: Args) {
      const due = runs
        .filter((run) => !reading.has(run.id))
        .filter(
          (run) =>
            (checked && !lastRead.has(run.id)) ||
            (failures.has(run.id) &&
              (failures.get(run.id) ?? 0) < MAX_FAILED_ATTEMPTS),
        )
        .sort(
          (a, b) =>
            (lastRead.has(a.id) ? 1 : 0) - (lastRead.has(b.id) ? 1 : 0) ||
            (lastRead.get(a.id) ?? 0) - (lastRead.get(b.id) ?? 0),
        )
        .slice(0, MAX_CHECKS_PER_LIST);
      await Promise.all(
        due.map((run) => readOne(run, { log, onRead, onGone })),
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
