import { HttpError } from "../../api/errors";
import { httpErrorFrom, type OpenResponse } from "../../api/http";
import { readSseFrames } from "../../api/sse";
import type { StreamEntry } from "./model/applyEntry";

/** No byte for this long means the connection is dead: routes heartbeat every 10 s. */
const LIVENESS_MS = 30_000;
const LIVENESS_CHECK_MS = 5_000;
/** The load balancer cuts an SSE response at 30 minutes; rotate before it does. */
const ROTATE_AFTER_MS = 25 * 60_000;

export type ConnectionResult =
  /** The body ended. `entries` counts stored entries; zero means a synthetic-only response. */
  | { kind: "ended"; entries: number; syntheticError: string | null }
  /** The route answered with an error status before streaming. */
  | { kind: "refused"; error: HttpError }
  /** Transport failure, liveness timeout, rotation or cancellation. */
  | { kind: "failed"; error: unknown };

/**
 * One SSE connection of a turn. Frames are handed over as they arrive; the
 * caller owns the turn state and decides whether and where to resume.
 */
export async function runConnection(args: {
  open: (signal: AbortSignal) => Promise<OpenResponse>;
  controller: AbortController;
  onAccepted: () => void;
  onEntry: (entry: StreamEntry) => void;
}): Promise<ConnectionResult> {
  const { controller } = args;
  let opened: OpenResponse;
  try {
    opened = await args.open(controller.signal);
  } catch (error) {
    return { kind: "failed", error };
  }
  const { response, release } = opened;
  const openedAt = Date.now();
  let lastActivity = openedAt;
  const watchdog = setInterval(() => {
    const now = Date.now();
    if (now - lastActivity > LIVENESS_MS || now - openedAt > ROTATE_AFTER_MS)
      controller.abort();
  }, LIVENESS_CHECK_MS);
  try {
    if (!response.ok)
      return {
        kind: "refused",
        error: await httpErrorFrom(response, controller.signal),
      };
    args.onAccepted();
    if (response.status === 204 || !response.body)
      return { kind: "ended", entries: 0, syntheticError: null };
    let entries = 0;
    let syntheticError: string | null = null;
    await readSseFrames(
      response.body,
      (frame) => {
        lastActivity = Date.now();
        if (frame.kind === "entry") {
          entries += 1;
          args.onEntry({
            turn: frame.turn,
            entryID: frame.entryId,
            chunk: frame.chunk,
          });
        } else if (frame.kind === "synthetic" && frame.chunk.type === "error") {
          syntheticError =
            typeof frame.chunk.errorText === "string"
              ? frame.chunk.errorText
              : "error";
        }
      },
      controller.signal,
    );
    if (controller.signal.aborted)
      return { kind: "failed", error: new Error("aborted") };
    return { kind: "ended", entries, syntheticError };
  } catch (error) {
    return { kind: "failed", error };
  } finally {
    clearInterval(watchdog);
    release();
  }
}

/** Last-applied stream activity, for deciding whether a foregrounded app must reconnect now. */
export function isStale(lastActivity: number, now = Date.now()): boolean {
  return now - lastActivity > LIVENESS_MS / 2;
}
