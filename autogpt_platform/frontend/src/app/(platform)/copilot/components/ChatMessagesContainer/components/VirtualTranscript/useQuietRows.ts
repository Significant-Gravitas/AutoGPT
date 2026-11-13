import { useLayoutEffect, useRef } from "react";
import type { TranscriptRow } from "./helpers";

/**
 * Which mounted rows skip the entrance animation. A row decides once, when it
 * mounts: rows that were already in the thread and only scroll back into the
 * window stay quiet, brand-new messages animate as before.
 */
export function useQuietRows({
  rows,
  indexes,
  isTracking,
}: {
  rows: TranscriptRow[];
  indexes: number[];
  /** False on the first pass, so every row of the opening window animates. */
  isTracking: boolean;
}) {
  const seenIDsRef = useRef(new Set<string>());
  const quietByIDRef = useRef(new Map<string, boolean>());
  const quietByID = quietByIDRef.current;
  for (const index of indexes) {
    const id = rows[index].message.id;
    if (!quietByID.has(id)) quietByID.set(id, seenIDsRef.current.has(id));
  }

  useLayoutEffect(() => {
    if (!isTracking) return;
    rememberMountedRows({
      rows,
      indexes,
      seenIDs: seenIDsRef.current,
      quietByID: quietByIDRef.current,
    });
  });

  return (id: string) => quietByID.get(id) ?? false;
}

/** Rows mounted from here on, other than brand-new messages, skip the
 *  entrance animation: they were already in the thread and are only
 *  scrolling back into the window. */
function rememberMountedRows({
  rows,
  indexes,
  seenIDs,
  quietByID,
}: {
  rows: TranscriptRow[];
  indexes: number[];
  seenIDs: Set<string>;
  quietByID: Map<string, boolean>;
}) {
  const mounted = new Set(indexes.map((index) => rows[index]?.message.id));
  for (const id of Array.from(quietByID.keys())) {
    if (!mounted.has(id)) quietByID.delete(id);
  }
  for (const row of rows) seenIDs.add(row.message.id);
}
