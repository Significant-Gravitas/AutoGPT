// Background upload queue for recording parts.
//
// The queue exists so the network is never on the critical path of the
// recording. Parts are already durable in IndexedDB by the time they get
// here, so a failure is a retry, not an error: nothing is surfaced to the
// user while they are still talking, and a dropped connection just parks
// the queue until the browser says it is back.

import { uploadBrainDumpPart } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useEffect, useRef, useState } from "react";
import { markPartUploaded, partId, RecordingPart } from "./recordingStore";

const RETRY_DELAYS_MS = [1000, 3000, 8000];
const FLUSH_PASSES = 3;

// 408 and 429 are the server asking us to come back. Every other 4xx is a
// verdict on the request itself — a part over the size cap, a body the
// server will not accept, an expired session — and retrying it produces
// the identical rejection three more times.
const RETRYABLE_CLIENT_STATUSES = [408, 429];

export function useUploadQueue() {
  const queueRef = useRef<RecordingPart[]>([]);
  const drainRef = useRef<Promise<void> | null>(null);
  const [pendingCount, setPendingCount] = useState(0);

  const [isOffline, setIsOffline] = useState(false);
  useEffect(() => {
    function handleOnline() {
      setIsOffline(false);
      void drain();
    }
    function handleOffline() {
      setIsOffline(true);
    }
    setIsOffline(typeof navigator !== "undefined" && !navigator.onLine);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Returns the IN-FLIGHT drain rather than a no-op when one is already
  // running. A bare `return` here resolves instantly, which made `flush()`
  // await nothing, observe the not-yet-uploaded tail and report failure —
  // on every single dump, since the final chunk is enqueued microseconds
  // before "I'm done" runs.
  function drain(): Promise<void> {
    if (drainRef.current) return drainRef.current;
    const running = drainQueue().finally(() => {
      drainRef.current = null;
    });
    drainRef.current = running;
    return running;
  }

  async function drainQueue() {
    while (queueRef.current.length > 0) {
      const part = queueRef.current[0];
      const uploaded = await uploadWithRetries(part);
      if (!uploaded) {
        // Leave it at the head — reconnecting replays from here so
        // parts still reach the server in order.
        return;
      }
      queueRef.current = queueRef.current.slice(1);
      setPendingCount(queueRef.current.length);
      // Bookkeeping only, and the part is already on the server. Where
      // IndexedDB is unavailable this rejects — and an unhandled
      // rejection here would take out `flush()` (reporting failure for a
      // dump that fully uploaded) and leave `enqueue()`'s `void drain()`
      // dangling.
      await markPartUploaded(part.id).catch(() => undefined);
    }
  }

  function enqueue(part: RecordingPart) {
    queueRef.current = [...queueRef.current, part];
    setPendingCount(queueRef.current.length);
    void drain();
  }

  // Called on "I'm done": every part must be at the server before
  // finalize runs, otherwise the assembled audio would have holes.
  //
  // Loops because a part can be enqueued in the gap between the running
  // drain emptying the queue and this check — one extra pass catches the
  // straggler, and the bound stops a pathological ping-pong.
  async function flush() {
    for (let attempt = 0; attempt < FLUSH_PASSES; attempt++) {
      await drain();
      if (queueRef.current.length === 0) return true;
    }
    return queueRef.current.length === 0;
  }

  function reset() {
    queueRef.current = [];
    setPendingCount(0);
  }

  return { enqueue, flush, reset, pendingCount, isOffline };
}

async function uploadWithRetries(part: RecordingPart) {
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
    try {
      await uploadBrainDumpPart({
        file: part.blob,
        recording_id: part.recordingId,
        part_index: part.partIndex,
      });
      return true;
    } catch (error) {
      // A part the server will never accept would otherwise be retried
      // for 12s here and then again on every 3-second chunk that follows,
      // holding the whole queue — and its blobs — at the head for the
      // rest of the take.
      if (isPermanentFailure(error)) return false;
      const delay = RETRY_DELAYS_MS[attempt];
      if (delay === undefined) return false;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  return false;
}

function isPermanentFailure(error: unknown) {
  if (!(error instanceof ApiError)) return false;
  return (
    error.status >= 400 &&
    error.status < 500 &&
    !RETRYABLE_CLIENT_STATUSES.includes(error.status)
  );
}

export function buildPart(
  recordingId: string,
  partIndex: number,
  blob: Blob,
): RecordingPart {
  return {
    id: partId(recordingId, partIndex),
    recordingId,
    partIndex,
    blob,
    savedAt: Date.now(),
    uploaded: false,
  };
}
