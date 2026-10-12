import { useEffect, useState } from "react";
import { BEAT_KEYS, type BeatKey } from "./flowItems";
import { PROMPT_DELAY_MS } from "./helpers";

type Beats = Record<BeatKey, boolean>;

function beats(value: boolean | Beats): Beats {
  if (typeof value !== "boolean") return { ...value };
  return BEAT_KEYS.reduce(
    (acc, key) => ({ ...acc, [key]: value }),
    {} as Beats,
  );
}

// Paces the stream: a question waits a beat after the answer before it, and
// the controls it introduces wait for it to finish typing. Seeded from the
// restored draft so a refresh mid-flow replays nothing.
export function useFlowProgress(triggers: Beats) {
  const [prompts, setPrompts] = useState<Beats>(() => beats(triggers));
  const [steps, setSteps] = useState<Beats>(() => beats(triggers));
  const pending = BEAT_KEYS.filter((key) => triggers[key] && !prompts[key]);
  const pendingKey = pending.join(",");

  useEffect(() => {
    if (!pendingKey) return;
    const timer = setTimeout(() => {
      setPrompts((current) => {
        const next = { ...current };
        pendingKey.split(",").forEach((key) => {
          next[key as BeatKey] = true;
        });
        return next;
      });
    }, PROMPT_DELAY_MS);
    return () => clearTimeout(timer);
  }, [pendingKey]);

  function revealStep(beat: BeatKey) {
    setSteps((current) => ({ ...current, [beat]: true }));
  }

  function reset() {
    setPrompts(beats(false));
    setSteps(beats(false));
  }

  // Prompts and steps latch on, so re-opening a beat has to unlatch the ones
  // that came after it.
  function clearAfter(beat: BeatKey) {
    const later = BEAT_KEYS.slice(BEAT_KEYS.indexOf(beat) + 1);
    const clear = (current: Beats) =>
      later.reduce((acc, key) => ({ ...acc, [key]: false }), { ...current });
    setPrompts(clear);
    setSteps(clear);
  }

  return { prompts, steps, revealStep, reset, clearAfter };
}
