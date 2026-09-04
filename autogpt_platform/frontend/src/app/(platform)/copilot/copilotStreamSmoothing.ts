import type { UIMessageChunk } from "ai";

/**
 * Client-side "typewriter" smoothing for the copilot SSE stream.
 *
 * The backend relays LLM output in irregular bursts (provider token batches +
 * Redis relay), so raw `text-delta` chunks make prose jump a phrase at a time.
 * This TransformStream re-chunks text/reasoning deltas into word-sized deltas
 * paced `TICK_DELAY_MS` apart so text flows steadily regardless of network
 * burst size.
 *
 * Pacing is adaptive: each tick emits ~1/`BACKLOG_DRAIN_TICKS` of the
 * buffered backlog (minimum one word), so output starts fast after a burst
 * and decelerates as it catches up — a 300-word burst drains in ~76 ticks
 * (<1 s) instead of 300.
 *
 * Ordering guarantees:
 *  - Any non-delta chunk (`text-end`, tool chunks, `finish`, …) flushes the
 *    pending buffer for the current part before passing through, so the
 *    stream envelope AI SDK's parser depends on is never reordered.
 *  - A delta for a different part id (or part type) also flushes first.
 *  - Deltas carrying `providerMetadata` are treated as barriers and passed
 *    through unsplit — splitting would either drop or duplicate the metadata.
 *
 * Only trailing partial words are held back between ticks; whitespace is
 * always emitted with the word preceding it, so Streamdown's
 * incomplete-markdown handling sees the same boundaries it would on the raw
 * stream.
 */

const TICK_DELAY_MS = 10;
const BACKLOG_DRAIN_TICKS = 25;

type SmoothableChunk = Extract<
  UIMessageChunk,
  { type: "text-delta" | "reasoning-delta" }
>;

interface PendingText {
  type: SmoothableChunk["type"];
  id: string;
  text: string;
}

function isSmoothable(chunk: UIMessageChunk): chunk is SmoothableChunk {
  return chunk.type === "text-delta" || chunk.type === "reasoning-delta";
}

function makeDelta(pending: PendingText, delta: string): UIMessageChunk {
  return pending.type === "text-delta"
    ? { type: "text-delta", id: pending.id, delta }
    : { type: "reasoning-delta", id: pending.id, delta };
}

/**
 * Cut points after each complete word (word + its trailing whitespace).
 * A trailing run of non-whitespace is a partial word and gets no cut point —
 * it stays buffered until more text or a flush arrives.
 */
function findWordCutPoints(text: string): number[] {
  const wordWithTrailingSpace = /\s*\S+\s+/gy;
  const cuts: number[] = [];
  while (wordWithTrailingSpace.exec(text)) {
    cuts.push(wordWithTrailingSpace.lastIndex);
  }
  return cuts;
}

function defaultWait(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

export function createSmoothingTransform(
  wait: (ms: number) => Promise<void> = defaultWait,
): TransformStream<UIMessageChunk, UIMessageChunk> {
  let pending: PendingText | null = null;

  function flushPending(
    controller: TransformStreamDefaultController<UIMessageChunk>,
  ) {
    if (pending && pending.text.length > 0) {
      controller.enqueue(makeDelta(pending, pending.text));
    }
    pending = null;
  }

  async function drainWords(
    controller: TransformStreamDefaultController<UIMessageChunk>,
  ) {
    while (pending) {
      const cuts = findWordCutPoints(pending.text);
      if (cuts.length === 0) return;
      const wordsThisTick = Math.max(
        1,
        Math.ceil(cuts.length / BACKLOG_DRAIN_TICKS),
      );
      const cutIndex = cuts[Math.min(wordsThisTick, cuts.length) - 1];
      controller.enqueue(makeDelta(pending, pending.text.slice(0, cutIndex)));
      pending.text = pending.text.slice(cutIndex);
      await wait(TICK_DELAY_MS);
    }
  }

  // The bundled lib.dom typings predate the Transformer `cancel` hook
  // (whatwg/streams#1283); extend the type until the TS lib catches up.
  const transformer: Transformer<UIMessageChunk, UIMessageChunk> & {
    cancel?: (reason: unknown) => void;
  } = {
    async transform(chunk, controller) {
      if (!isSmoothable(chunk) || chunk.providerMetadata) {
        flushPending(controller);
        controller.enqueue(chunk);
        return;
      }
      if (pending && (pending.id !== chunk.id || pending.type !== chunk.type)) {
        flushPending(controller);
      }
      if (!pending) {
        pending = { type: chunk.type, id: chunk.id, text: "" };
      }
      pending.text += chunk.delta;
      await drainWords(controller);
    },
    flush(controller) {
      flushPending(controller);
    },
    // Reader cancellation (stop button, unmount) tears the stream down —
    // nothing can be delivered after it, so drop the buffer instead of
    // flushing and let any in-flight drain loop exit without enqueueing
    // into the cancelled stream.
    cancel() {
      pending = null;
    },
  };

  return new TransformStream<UIMessageChunk, UIMessageChunk>(transformer);
}
