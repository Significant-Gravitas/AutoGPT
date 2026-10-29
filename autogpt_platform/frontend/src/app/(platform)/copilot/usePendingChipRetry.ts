import * as Sentry from "@sentry/nextjs";
import { useRef } from "react";

import type { QueuedMessage } from "./helpers/pendingChipQueue";
import { resendMetadata, type ResendMetadata } from "./helpers/resendMetadata";

/** Sends a follow-up again as a turn of its own, the way the chat sends. */
export type ResendFollowUp = (
  text: string,
  metadata: ResendMetadata | undefined,
) => Promise<unknown>;

interface Args {
  sessionId: string | null;
  queue: QueuedMessage[];
  takeQueuedMessage: (id: string) => void;
  resend: ResendFollowUp;
  /** False leaves the chip as it is (the chat can't send right now). */
  canRetry?: () => boolean;
}

/**
 * Retry for a follow-up the last turn ended without reading. The chip leaves
 * the strip and the follow-up goes out by its id: the server takes that very
 * message off the queue and runs it as a turn, or runs nothing if a turn read
 * it meanwhile, so Otto never reads it twice. One at a time: the turn a
 * retry starts reads every other held follow-up.
 */
export function usePendingChipRetry({
  sessionId,
  queue,
  takeQueuedMessage,
  resend,
  canRetry,
}: Args) {
  const isRetryingRef = useRef(false);

  async function retry(index: number) {
    const entry = queue[index];
    if (!entry || !sessionId || isRetryingRef.current) return;
    if (canRetry && !canRetry()) return;
    isRetryingRef.current = true;
    takeQueuedMessage(entry.id);
    const metadata = entry.serverId
      ? resendMetadata(entry.serverId)
      : undefined;
    await resend(entry.text, metadata).catch((err: unknown) =>
      Sentry.captureException(err),
    );
    isRetryingRef.current = false;
  }

  return retry;
}
