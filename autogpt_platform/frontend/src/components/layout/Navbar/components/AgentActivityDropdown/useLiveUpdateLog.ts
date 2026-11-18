import { getGetV1ListAllExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { hashKey, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

const MAX_TRACKED_LISTS = 10;
// A retry starts no earlier than the failure before it ("failed"), and a
// fetch paused while offline starts when it continues; neither dispatches
// another "fetch".
const ATTEMPT_STARTS = new Set(["fetch", "failed", "continue"]);

type RunStatus = GraphExecution | GraphExecutionMeta;

/**
 * Orders what Activity learns about runs, so rebuilding from a list keeps
 * what it can't know yet. The backend writes a change before it publishes
 * it, so a fetch that started after an update arrived has read that change
 * or a newer one, while an update that arrived after a fetch started may be
 * newer than what the fetch read. Each update, each fetch start and each
 * read of a single run takes the next number in one sequence, which tells
 * them apart.
 */
export function useLiveUpdateLog() {
  const queryClient = useQueryClient();
  const [log] = useState(createLiveUpdateLog);

  useEffect(() => {
    const listHash = hashKey(getGetV1ListAllExecutionsQueryKey());
    return queryClient.getQueryCache().subscribe((event) => {
      if (event.type !== "updated" || event.query.queryHash !== listHash)
        return;
      if (ATTEMPT_STARTS.has(event.action.type)) log.recordFetchStart();
      if (event.action.type === "success") {
        log.recordFetchSuccess(event.query.state.dataUpdatedAt);
      }
    });
  }, [log, queryClient]);

  return log;
}

export function createLiveUpdateLog() {
  let sequence = 0;
  let fetchStartSequence = 0;
  const updates = new Map<string, { execution: RunStatus; sequence: number }>();
  // dataUpdatedAt of a list -> sequence when the fetch behind it started
  const listSequences = new Map<number, number>();
  // run id -> its read by ID under way: when it started, and whether newer
  // news of the run has come since (a live update, or a list fetched later)
  const reads = new Map<string, { started: number; superseded: boolean }>();

  return {
    recordUpdate(execution: GraphExecution) {
      sequence += 1;
      updates.delete(execution.id);
      updates.set(execution.id, { execution, sequence });
      const read = reads.get(execution.id);
      if (read) read.superseded = true;
    },
    /** A read of one run by ID starts now. */
    startRead(runID: string) {
      sequence += 1;
      reads.set(runID, { started: sequence, superseded: false });
    },
    /**
     * A read by ID has settled. True if nothing newer about the run came
     * meanwhile. Its answer then stands for the moment the read started, so
     * a list fetched before that moment replays it.
     */
    finishRead(runID: string, read?: RunStatus) {
      const pending = reads.get(runID);
      reads.delete(runID);
      if (!pending || pending.superseded) return false;
      if (read) {
        updates.delete(runID);
        updates.set(runID, { execution: read, sequence: pending.started });
      }
      return true;
    },
    recordFetchStart() {
      sequence += 1;
      fetchStartSequence = sequence;
    },
    recordFetchSuccess(dataUpdatedAt: number) {
      listSequences.set(dataUpdatedAt, fetchStartSequence);
      if (listSequences.size > MAX_TRACKED_LISTS) {
        const oldest = listSequences.keys().next().value;
        if (oldest !== undefined) listSequences.delete(oldest);
      }
    },
    /**
     * What the list fetched at `dataUpdatedAt` may predate, in arrival
     * order. Forgets the rest, which every later list includes. A read by ID
     * gives way to a list that started later and either has the run or has
     * every run (`complete`), so the run is gone. A list whose fetch start
     * went unseen counts as older than everything.
     */
    updatesNewerThan(
      dataUpdatedAt: number,
      listed: { id: string }[],
      complete: boolean,
    ) {
      const since = listSequences.get(dataUpdatedAt) ?? 0;
      const listedIDs = new Set(listed.map(({ id }) => id));
      reads.forEach((read, id) => {
        if (since > read.started && (complete || listedIDs.has(id))) {
          read.superseded = true;
        }
      });
      const newer: RunStatus[] = [];
      updates.forEach((update, id) => {
        if (update.sequence > since) newer.push(update.execution);
        else updates.delete(id);
      });
      return newer;
    },
  };
}
