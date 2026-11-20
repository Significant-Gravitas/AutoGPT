import {
  getGetV1ListAllExecutionsQueryKey,
  getV1ListAllExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { hashKey, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { isActiveExecution } from "./helpers";

const MAX_TRACKED_LISTS = 10;
// Activity keeps a finished run only while its end is within 72 hours (see
// categorizeExecutions in helpers.tsx): entries for runs no list covers
// must not pile up past that.
const MILLISECONDS_PER_72_HOURS = 72 * 60 * 60 * 1000;

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
  const [log] = useState(() => sharedLiveUpdateLog);

  useEffect(() => {
    const listHash = hashKey(getGetV1ListAllExecutionsQueryKey());
    return queryClient.getQueryCache().subscribe((event) => {
      if (event.type !== "updated" || event.query.queryHash !== listHash)
        return;
      // Attempt starts are recorded by the shared query function, which
      // runs on every attempt including retries (a retry dispatches no
      // "fetch" action, so the cache events can't time it). Only the
      // success lands here, to map the data to its fetch start.
      if (event.action.type === "success") {
        log.recordFetchSuccess(event.query.state.dataUpdatedAt);
      }
    });
  }, [log, queryClient]);

  return log;
}

// One log for the app: the five hooks that read the run list share one
// query function (below), and it records each attempt's start in this log,
// so a fetch uses the same log whichever observer's options were set last.
const sharedLiveUpdateLog = createLiveUpdateLog();

/** The shared query function for the run list, for every hook that reads it. */
export function listAllExecutionsQueryFn({ signal }: { signal?: AbortSignal }) {
  sharedLiveUpdateLog.recordFetchStart();
  return getV1ListAllExecutions({ signal });
}

/** For tests that render the dropdown: drop the shared log's state. */
export function resetSharedLiveUpdateLog() {
  sharedLiveUpdateLog.reset();
}

export function createLiveUpdateLog() {
  let sequence = 0;
  let fetchStartSequence = 0;
  const updates = new Map<string, { execution: RunStatus; sequence: number }>();
  // dataUpdatedAt of a list -> sequence when the fetch behind it started
  const listSequences = new Map<number, number>();
  // run id -> its read by ID under way: when it started, and whether newer
  // news of the run has come since (a live update, or a list fetched later),
  // with that news's place in the sequence, so a queued fresh read can tell
  // whether the news also covers it.
  const reads = new Map<
    string,
    { started: number; superseded: boolean; supersededAt?: number }
  >();

  return {
    /** Drop everything: for tests that share the log across renders. */
    reset() {
      sequence = 0;
      fetchStartSequence = 0;
      updates.clear();
      listSequences.clear();
      reads.clear();
    },
    /** The log's current ordering number, for dating a queued check. */
    currentSequence() {
      return sequence;
    },
    recordUpdate(execution: GraphExecution) {
      sequence += 1;
      updates.delete(execution.id);
      updates.set(execution.id, { execution, sequence });
      const read = reads.get(execution.id);
      if (read) {
        read.superseded = true;
        read.supersededAt = sequence;
      }
    },
    /** A read of one run by ID starts now. */
    startRead(runID: string) {
      sequence += 1;
      reads.set(runID, { started: sequence, superseded: false });
    },
    /**
     * A read by ID has settled. True if nothing newer about the run came
     * meanwhile. Its answer then stands for the moment the read started, so
     * a list fetched before that moment replays it. When newer news did
     * come, `supersededAt` says where in the sequence it sits, so the
     * caller can tell whether it also covers a check queued after the read
     * started; unknown when the read is gone (a reset dropped it).
     */
    finishRead(runID: string, read?: RunStatus) {
      const pending = reads.get(runID);
      reads.delete(runID);
      if (!pending || pending.superseded) {
        return { accepted: false, supersededAt: pending?.supersededAt };
      }
      if (read) {
        updates.delete(runID);
        updates.set(runID, { execution: read, sequence: pending.started });
      }
      return { accepted: true };
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
     * went unseen counts as older than everything. A capped list that
     * leaves a run out says nothing about it, so its entry is replayed over
     * the list instead of dropped — unless the run is past Activity's
     * 72-hour retention.
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
          // The list's fetch started here: that is where its observation
          // sits in the sequence.
          read.supersededAt = since;
        }
      });
      const newer: RunStatus[] = [];
      updates.forEach((update, id) => {
        if (update.sequence > since) {
          newer.push(update.execution);
        } else if (complete || listedIDs.has(id)) {
          // The list's fetch started after this update, and the list has
          // the run (or has every run): the list already includes it.
          updates.delete(id);
        } else if (!isStillRecent(update.execution)) {
          updates.delete(id);
        } else {
          // A capped list that leaves the run out says nothing about it,
          // so the entry is not in the list: replay it over the list.
          newer.push(update.execution);
        }
      });
      return newer;
    },
  };
}

// Whether Activity would still show this run: active runs always, and a
// finished run only while its end is within 72 hours. A run with no end
// recorded stays, since nothing proves it stale.
function isStillRecent(execution: RunStatus): boolean {
  if (isActiveExecution(execution)) return true;
  if (execution.ended_at === undefined || execution.ended_at === null) {
    return true;
  }
  return (
    new Date(execution.ended_at).getTime() >
    Date.now() - MILLISECONDS_PER_72_HOURS
  );
}
