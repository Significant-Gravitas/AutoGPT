/**
 * Synthesises chunks ahead of playback and plays them in order.
 *
 * One `<audio>` element for the whole session: browsers only grant autoplay
 * to an element first played inside a user gesture, and every later chunk
 * arrives from the network with no gesture of its own.
 */

const SILENT_WAV =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=";

interface Args {
  synthesize: (text: string) => Promise<Blob>;
  /** The queue ran dry. The caller decides whether the reply is over. */
  onIdle: () => void;
  onError: (error: unknown) => void;
}

export function createSpeechPlayer({ synthesize, onIdle, onError }: Args) {
  const queue: Promise<Blob | null>[] = [];
  let element: HTMLAudioElement | null = null;
  let draining = false;
  /** Bumped by `stop`, so audio synthesised for an abandoned turn is dropped. */
  let generation = 0;

  return { unlock, enqueue, stop, destroy, isIdle };

  /** Must be called from a click handler, before the first `enqueue`. */
  function unlock() {
    if (element) return;
    element = new Audio(SILENT_WAV);
    element.preload = "auto";
    void element.play().catch(() => undefined);
  }

  function enqueue(text: string) {
    queue.push(
      synthesize(text).catch((error) => {
        onError(error);
        return null;
      }),
    );
    void drain();
  }

  async function drain() {
    if (draining) return;
    draining = true;
    while (queue.length > 0) {
      const mine = generation;
      const blob = await queue.shift();
      if (blob && mine === generation) await play(blob);
    }
    draining = false;
    onIdle();
  }

  async function play(blob: Blob) {
    unlock();
    const url = URL.createObjectURL(blob);
    try {
      await playUrl(url);
    } catch (error) {
      onError(error);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function playUrl(url: string): Promise<void> {
    const audio = element;
    if (!audio) return Promise.resolve();
    audio.src = url;
    return new Promise((resolve, reject) => {
      function settle(finish: () => void) {
        audio!.removeEventListener("ended", onEnded);
        audio!.removeEventListener("error", onFailed);
        finish();
      }
      function onEnded() {
        settle(resolve);
      }
      function onFailed() {
        settle(() => reject(new Error("Audio playback failed")));
      }
      audio.addEventListener("ended", onEnded);
      audio.addEventListener("error", onFailed);
      audio.play().catch((error) => settle(() => reject(error)));
    });
  }

  /** Abandon everything queued and cut off what is playing. */
  function stop() {
    generation += 1;
    queue.length = 0;
    if (element) {
      element.pause();
      element.src = SILENT_WAV;
    }
  }

  function destroy() {
    stop();
    element = null;
  }

  function isIdle() {
    return !draining && queue.length === 0;
  }
}

export type SpeechPlayer = ReturnType<typeof createSpeechPlayer>;
