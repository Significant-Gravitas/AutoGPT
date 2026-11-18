import { getGetV1ListAllExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

const MAX_TRACKED_LISTS = 10;

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
    const [listKey] = getGetV1ListAllExecutionsQueryKey();
    return queryClient.getQueryCache().subscribe((event) => {
      if (event.type !== "updated" || event.query.queryKey[0] !== listKey)
        return;
      if (event.action.type === "fetch") log.recordFetchStart();
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

  return {
    recordUpdate(execution: GraphExecution) {
      sequence += 1;
      updates.delete(execution.id);
      updates.set(execution.id, { execution, sequence });
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
