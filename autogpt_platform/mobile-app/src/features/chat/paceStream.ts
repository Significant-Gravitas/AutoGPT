import type { UIMessageChunk } from "ai";

const TICK_MS = 24;
// Each tick lets out this fraction of what waits, so a steady stream shows
// about ten ticks (a quarter second) behind the network and a burst eases out.
const DRAIN_TICKS = 10;
const MIN_CHARS = 3;
// How far past the budget a cut may reach to end on a space.
const WORD_SLACK = 12;

interface PaceOptions {
  /** Called once, when the first words go out. */
  onFirstText?: () => void;
}

/**
 * The network hands over text in uneven bursts. This lets it out at a steady
 * pace, a few words at a time, with every other chunk kept in order.
 */
export function paceTextStream(
  source: ReadableStream<UIMessageChunk>,
  { onFirstText }: PaceOptions = {},
) {
  const reader = source.getReader();
  const queue: UIMessageChunk[] = [];
  let isSourceDone = false;
  let hasText = false;
  let timer: ReturnType<typeof setInterval> | null = null;

  function stopTimer() {
    if (timer !== null) clearInterval(timer);
    timer = null;
  }

  return new ReadableStream<UIMessageChunk>({
    start(controller) {
      function emit(chunks: UIMessageChunk[]) {
        for (const chunk of chunks) {
          if (!hasText && chunk.type === "text-delta") {
            hasText = true;
            onFirstText?.();
          }
          controller.enqueue(chunk);
        }
      }

      timer = setInterval(() => {
        emit(takeTick(queue));
        if (isSourceDone && queue.length === 0) {
          stopTimer();
          controller.close();
        }
      }, TICK_MS);

      void (async () => {
        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            queue.push(value);
          }
          isSourceDone = true;
        } catch (error) {
          // What already arrived still shows; then the SDK sees the failure.
          stopTimer();
          try {
            emit(queue.splice(0));
            controller.error(error);
          } catch {
            // The reader already cancelled this stream.
          }
        }
      })();
    },
    cancel(reason) {
      stopTimer();
      queue.length = 0;
      return reader.cancel(reason);
    },
  });
}

/** Takes one tick's worth of chunks off the front of the queue. */
export function takeTick(queue: UIMessageChunk[]): UIMessageChunk[] {
  let budget = Math.max(MIN_CHARS, Math.ceil(pendingChars(queue) / DRAIN_TICKS));
  const out: UIMessageChunk[] = [];
  while (queue.length > 0) {
    const head = queue[0];
    if (head.type !== "text-delta") {
      out.push(head);
      queue.shift();
      continue;
    }
    if (budget <= 0) break;
    const cut = cutPoint(head.delta, budget);
    const piece = head.delta.slice(0, cut);
    budget -= piece.length;
    if (cut >= head.delta.length) queue.shift();
    else queue[0] = { ...head, delta: head.delta.slice(cut) };
    const last = out[out.length - 1];
    if (last?.type === "text-delta" && last.id === head.id) {
      out[out.length - 1] = { ...last, delta: last.delta + piece };
    } else {
      out.push({ ...head, delta: piece });
    }
  }
  return out;
}

function pendingChars(queue: UIMessageChunk[]) {
  let total = 0;
  for (const chunk of queue) {
    if (chunk.type === "text-delta") total += chunk.delta.length;
  }
  return total;
}

/** Where to cut `text` near `budget`: after a space if one is close, never inside a surrogate pair. */
export function cutPoint(text: string, budget: number) {
  if (text.length <= budget) return text.length;
  const space = text.slice(budget, budget + WORD_SLACK).search(/\s/);
  let cut = space === -1 ? budget : budget + space + 1;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) cut += 1;
  return Math.min(cut, text.length);
}
