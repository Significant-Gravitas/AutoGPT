import type { UIMessageChunk } from "ai";

import { findSegmentCutPoints } from "@/lib/text-segments";
import {
  createTextPacer,
  defaultNow,
  defaultTickMs,
  defaultWait,
  isDocumentHidden,
} from "./copilotStreamPacer";

/**
 * Client-side "typewriter" smoothing for the copilot SSE stream.
 *
 * The backend relays LLM output in irregular bursts (provider token batches,
 * the Redis relay, sometimes a whole text block in one delta), so raw
 * `text-delta` chunks make prose jump a phrase at a time. This
 * TransformStream re-chunks text/reasoning deltas into word-sized deltas
 * revealed by elapsed time (`copilotStreamPacer.ts`): a steady rate, at most
 * `MAX_LAG_MS` behind what has arrived. Words are split by
 * `findSegmentCutPoints`, which also cuts Chinese, Japanese and Thai text,
 * written without spaces.
 *
 * Ordering guarantees:
 *  - Any non-delta chunk (`text-end`, tool chunks, …) passes through only
 *    after the pending buffer for the current part has been emitted, so the
 *    stream envelope AI SDK's parser depends on is never reordered. The
 *    buffer drains at its pace first; `finish`, `error` and `abort` flush it
 *    at once.
 *  - A delta for a different part id (or part type) also drains first.
 *  - Deltas carrying `providerMetadata` are treated as barriers and passed
 *    through unsplit — splitting would either drop or duplicate the metadata.
 *
 * Only a trailing partial word is held back between ticks; whitespace is
 * always emitted with the word preceding it, so Streamdown's
 * incomplete-markdown handling sees the same boundaries it would on the raw
 * stream. In a hidden tab, and when the stream ends, everything is emitted at
 * once.
 */

export interface SmoothingOptions {
  wait?: (ms: number) => Promise<void>;
  now?: () => number;
  isHidden?: () => boolean;
  tickMs?: number;
  /**
   * Pass chunks through unpaced until the input first goes quiet for this
   * long. A resumed stream replays the turn so far in one burst; that is
   * history, and only what arrives after it is paced.
   */
  replayUntilQuietMs?: number;
}

type SmoothableChunk = Extract<
  UIMessageChunk,
  { type: "text-delta" | "reasoning-delta" }
>;

interface PendingText {
  type: SmoothableChunk["type"];
  id: string;
  text: string;
}

const STREAM_END_TYPES = new Set(["finish", "error", "abort"]);

function isSmoothable(chunk: UIMessageChunk): chunk is SmoothableChunk {
  return chunk.type === "text-delta" || chunk.type === "reasoning-delta";
}

function makeDelta(pending: PendingText, delta: string): UIMessageChunk {
  return pending.type === "text-delta"
    ? { type: "text-delta", id: pending.id, delta }
    : { type: "reasoning-delta", id: pending.id, delta };
}

export function createSmoothingTransform(
  options: SmoothingOptions = {},
): TransformStream<UIMessageChunk, UIMessageChunk> {
  const wait = options.wait ?? defaultWait;
  const now = options.now ?? defaultNow;
  const isHidden = options.isHidden ?? isDocumentHidden;
  const tickMs = options.tickMs ?? defaultTickMs();
  const quietMs = options.replayUntilQuietMs;
  const pacer = createTextPacer();
  let pending: PendingText | null = null;
  let draining: Promise<void> | null = null;
  let closed = false;
  let replaying = quietMs !== undefined;
  let lastArrival: number | null = null;

  function emitUpTo(
    controller: TransformStreamDefaultController<UIMessageChunk>,
    cut: number,
  ) {
    if (!pending || cut <= 0) return;
    controller.enqueue(makeDelta(pending, pending.text.slice(0, cut)));
    pending.text = pending.text.slice(cut);
  }

  function flushPending(
    controller: TransformStreamDefaultController<UIMessageChunk>,
  ) {
    if (pending && pending.text.length > 0) {
      controller.enqueue(makeDelta(pending, pending.text));
    }
    pending = null;
  }

  // Emits what is due; true while complete words are still waiting.
  function emitDue(
    controller: TransformStreamDefaultController<UIMessageChunk>,
  ) {
    if (!pending) return false;
    const cuts = findSegmentCutPoints(pending.text);
    if (cuts.length === 0) return false;
    const last = cuts[cuts.length - 1];
    const cut = isHidden() ? last : pacer.take(cuts, now());
    emitUpTo(controller, cut);
    return cut < last;
  }

  async function drain(
    controller: TransformStreamDefaultController<UIMessageChunk>,
  ) {
    try {
      pacer.begin(now());
      do {
        await wait(tickMs);
      } while (!closed && emitDue(controller));
    } catch {
      // The readable side is gone; nothing can be delivered any more.
      closed = true;
    } finally {
      draining = null;
    }
  }

  // Paces out the complete words, then the partial one, before a barrier.
  async function settle(
    controller: TransformStreamDefaultController<UIMessageChunk>,
  ) {
    while (draining && !isHidden()) await draining;
    flushPending(controller);
  }

  // Replayed history passes straight through until the first quiet gap.
  function isReplayed(at: number) {
    if (!replaying || quietMs === undefined) return false;
    if (lastArrival !== null && at - lastArrival >= quietMs) replaying = false;
    lastArrival = at;
    return replaying;
  }

  // The bundled lib.dom typings predate the Transformer `cancel` hook
  // (whatwg/streams#1283); extend the type until the TS lib catches up.
  const transformer: Transformer<UIMessageChunk, UIMessageChunk> & {
    cancel?: (reason: unknown) => void;
  } = {
    async transform(chunk, controller) {
      if (isReplayed(now())) {
        controller.enqueue(chunk);
        return;
      }
      if (!isSmoothable(chunk) || chunk.providerMetadata) {
        if (STREAM_END_TYPES.has(chunk.type)) flushPending(controller);
        else await settle(controller);
        controller.enqueue(chunk);
        return;
      }
      if (pending && (pending.id !== chunk.id || pending.type !== chunk.type)) {
        await settle(controller);
      }
      if (!pending) {
        pending = { type: chunk.type, id: chunk.id, text: "" };
      }
      pending.text += chunk.delta;
      pacer.arrive(chunk.delta.length, now());
      if (isHidden()) {
        // A background tab's timers are throttled; show what is complete.
        emitDue(controller);
        return;
      }
      if (!draining) draining = drain(controller);
    },
    flush(controller) {
      flushPending(controller);
      closed = true;
    },
    // Reader cancellation (stop button, unmount) tears the stream down —
    // nothing can be delivered after it, so drop the buffer instead of
    // flushing and let any in-flight drain loop exit without enqueueing
    // into the cancelled stream.
    cancel() {
      pending = null;
      closed = true;
    },
  };

  return new TransformStream<UIMessageChunk, UIMessageChunk>(transformer);
}
