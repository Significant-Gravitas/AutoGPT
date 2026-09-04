/**
 * Synthesises chunks ahead of playback and plays them in order, through one
 * `<audio>` element: browsers only grant autoplay to an element first played
 * inside a user gesture, and later chunks arrive with no gesture of their own.
 */

import type { SpeechKind } from "./speechApi";

const SILENT_WAV =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=";

/**
 * One element for the tab, not for the hook: creating a session re-keys the
 * chat host, and an element first played before that remount is the only one
 * still allowed to play after it.
 */
let sharedAudio: HTMLAudioElement | null = null;

/** Call from a click. Safe to repeat; only the first play grants autoplay. */
export function unlockAudio(): HTMLAudioElement {
  if (!sharedAudio) {
    sharedAudio = new Audio(SILENT_WAV);
    sharedAudio.preload = "auto";
    void sharedAudio.play().catch(() => undefined);
  }
  return sharedAudio;
}

interface Args {
  synthesize: (text: string, kind: SpeechKind) => Promise<Blob>;
  /** The queue ran dry. The caller decides whether the reply is over. */
  onIdle: () => void;
  onError: (error: unknown) => void;
}

export function createSpeechPlayer({ synthesize, onIdle, onError }: Args) {
  const queue: Promise<Blob | null>[] = [];
  let draining = false;
  /** Bumped by `stop`, so audio synthesised for an abandoned turn is dropped. */
  let generation = 0;

  return { unlock, enqueue, stop, destroy, isIdle };

  /** Must be called from a click handler, before the first `enqueue`. */
  function unlock() {
    unlockAudio();
  }

  function enqueue(text: string, kind: SpeechKind = "reply") {
    queue.push(
      synthesize(text, kind).catch((error) => {
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
    const audio = unlockAudio();
    audio.src = url;
    return new Promise((resolve, reject) => {
      function settle(finish: () => void) {
        audio.removeEventListener("ended", onEnded);
        audio.removeEventListener("error", onFailed);
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
    if (sharedAudio) {
      sharedAudio.pause();
      sharedAudio.src = SILENT_WAV;
    }
  }

  function destroy() {
    stop();
  }

  function isIdle() {
    return !draining && queue.length === 0;
  }
}

export type SpeechPlayer = ReturnType<typeof createSpeechPlayer>;
