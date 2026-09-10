/**
 * One Web Audio context for the tab, shared by the click and by the analyser
 * that watches AutoPilot's own speech.
 *
 * Shared deliberately: a context created outside a user gesture starts
 * suspended, and a suspended context that has an `<audio>` element routed
 * through it plays nothing at all. One context, primed inside the toggle
 * click, is the only arrangement where both stay audible.
 */

type AudioContextCtor = new () => AudioContext;

let context: AudioContext | null = null;

/** Call from a click. Safe to repeat. */
export function primeAudioContext(): AudioContext | null {
  const ctor =
    typeof window === "undefined"
      ? null
      : ((window.AudioContext ?? null) as AudioContextCtor | null);
  if (!ctor) return null;
  try {
    context ??= new ctor();
    if (context.state === "suspended") void context.resume();
    return context;
  } catch {
    context = null;
    return null;
  }
}

/** The context only if it can actually make sound right now. */
export function runningAudioContext(): AudioContext | null {
  const ctx = primeAudioContext();
  return ctx?.state === "running" ? ctx : null;
}

/** Test seam: jsdom has no Web Audio, and a real one leaks between suites. */
export function resetAudioContext() {
  void context?.close().catch(() => undefined);
  context = null;
}
