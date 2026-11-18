import { getGetV1ListAllExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { hashKey, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

const MAX_TRACKED_LISTS = 10;
const MAX_TRACKED_RUNS = 1000;
// A retry starts no earlier than the failure before it ("failed"), and a
// fetch paused while offline starts when it continues; neither dispatches
// another "fetch".
const ATTEMPT_STARTS = new Set(["fetch", "failed", "continue"]);

/**
 * Orders live run updates against fetches of the run list, so rebuilding
 * from a list keeps what it can't know yet. The backend writes a change
 * before it publishes it, so a fetch that started after an update arrived
 * has read that change or a newer one, while an update that arrived after a
 * fetch started may be newer than what the fetch read. Each update and each
 * fetch start takes the next number in one sequence, which tells them apart.
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

function createLiveUpdateLog() {
  let sequence = 0;
  let fetchStartSequence = 0;
  const updates = new Map<
    string,
    { execution: GraphExecution; sequence: number }
  >();
  // dataUpdatedAt of a list -> sequence when the fetch behind it started
  const listSequences = new Map<number, number>();
  // run id -> sequence of its latest update; outlives the list rebuilds,
  // which forget `updates`, for the checks of single runs
  const latest = new Map<string, number>();

  return {
    recordUpdate(execution: GraphExecution) {
      sequence += 1;
      updates.delete(execution.id);
      updates.set(execution.id, { execution, sequence });
      latest.delete(execution.id);
      latest.set(execution.id, sequence);
      if (latest.size > MAX_TRACKED_RUNS) {
        const oldest = latest.keys().next().value;
        if (oldest !== undefined) latest.delete(oldest);
      }
    },
    /** Where the sequence stands, for a read of a single run starting now. */
    mark() {
      return sequence;
    },
    /** Whether a live update for the run arrived after `mark`. */
    updatedSince(runID: string, mark: number) {
      return (latest.get(runID) ?? 0) > mark;
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
     * The updates the list fetched at `dataUpdatedAt` may predate, in
     * arrival order. Forgets the rest, which every later list includes. A
     * list whose fetch start went unseen counts as older than every update.
     */
    updatesNewerThan(dataUpdatedAt: number) {
      const since = listSequences.get(dataUpdatedAt) ?? 0;
      const newer: GraphExecution[] = [];
      updates.forEach((update, id) => {
        if (update.sequence > since) newer.push(update.execution);
        else updates.delete(id);
      });
      return newer;
    },
  };
}
