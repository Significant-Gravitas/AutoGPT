import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  BRAIN_DUMP_PROMPT_SETS,
  type PromptSetName,
  promptSetNameForRole,
} from "./prompts";
import type { ScreenState } from "./useBrainDumpStep";
import { useIsWideLayout } from "./useIsWideLayout";

// Half way round the set, so the live prompt never repeats the first card
// the ambient layer shows.
const FIRST_INDEX = 4;

// Where the live prompt is on screen, and how long each question stays.
// At rest it only shows where the ambient cards do not fit.
const AUTO_ADVANCE_MS = {
  rest: 12_000,
  typing: 20_000,
  recording: 15_000,
} as const;

type PromptScreen = keyof typeof AUTO_ADVANCE_MS;
type AdvanceTrigger = "auto" | "click";

interface Args {
  role: string;
  screen: ScreenState;
}

export function usePromptRotation({ role, screen }: Args) {
  const promptSet = promptSetNameForRole(role);
  const hasAmbientCards = useIsWideLayout();
  const reduceMotion = useReducedMotion() === true;
  const [rotation, setRotation] = useState({ promptSet, index: FIRST_INDEX });
  const index = rotation.promptSet === promptSet ? rotation.index : FIRST_INDEX;
  const shownOn = livePromptScreen(screen, hasAmbientCards);
  const seenRef = useRef(new Set<string>());

  useEffect(() => {
    if (shownOn) seenRef.current.add(`${promptSet}:${index}`);
  }, [shownOn, promptSet, index]);

  // Restarted on every advance, so a click buys the new question its full
  // time on screen.
  useEffect(() => {
    if (!shownOn || reduceMotion) return;
    const timer = setInterval(() => {
      if (document.visibilityState !== "visible") return;
      setRotation(advanceFrom({ promptSet, index, screen: shownOn }, "auto"));
    }, AUTO_ADVANCE_MS[shownOn]);
    return () => clearInterval(timer);
  }, [shownOn, reduceMotion, promptSet, index]);

  function advance(trigger: AdvanceTrigger) {
    if (!shownOn) return;
    setRotation(advanceFrom({ promptSet, index, screen: shownOn }, trigger));
  }

  return {
    question: BRAIN_DUMP_PROMPT_SETS[promptSet].questions[index],
    index,
    promptSet,
    advance,
    // Read when an event is sent, never while rendering.
    promptsSeen: () => seenRef.current.size,
  };
}

function livePromptScreen(
  screen: ScreenState,
  hasAmbientCards: boolean,
): PromptScreen | null {
  if (screen === "typing" || screen === "recording") return screen;
  if (screen === "rest" && !hasAmbientCards) return screen;
  return null;
}

function advanceFrom(
  from: { promptSet: PromptSetName; index: number; screen: PromptScreen },
  trigger: AdvanceTrigger,
) {
  const { questions } = BRAIN_DUMP_PROMPT_SETS[from.promptSet];
  const next = (from.index + 1) % questions.length;
  trackBrainDump("brain_dump_prompt_advanced", {
    trigger,
    screen: from.screen,
    prompt_index: next,
    prompt_set: from.promptSet,
  });
  return { promptSet: from.promptSet, index: next };
}
