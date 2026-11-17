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

// Where the live prompt is on screen. At rest it only shows where the
// ambient cards do not fit.
type PromptScreen = "rest" | "typing" | "recording";
type AdvanceVia = "auto" | "click";

// How long each question stays. It keeps moving while recording, so a
// speaker who runs dry has something new to pick up; it is never announced,
// so it cannot talk into the take, and the user can pause it (togglePause).
const AUTO_ADVANCE_MS: Record<PromptScreen, number> = {
  rest: 12_000,
  typing: 20_000,
  recording: 15_000,
};

interface Args {
  role: string;
  screen: ScreenState;
}

export function usePromptRotation({ role, screen }: Args) {
  const promptSet = promptSetNameForRole(role);
  const hasAmbientCards = useIsWideLayout();
  const reduceMotion = useReducedMotion() === true;
  const [rotation, setRotation] = useState<Rotation>(firstRotation(promptSet));
  // One switch for every idea that changes on its own: the live prompt and
  // the ambient cards (WCAG 2.2.2).
  const [isPaused, setIsPaused] = useState(false);
  const current =
    rotation.promptSet === promptSet ? rotation : firstRotation(promptSet);
  const { index } = current;
  const shownOn = livePromptScreen(screen, hasAmbientCards);
  const question = BRAIN_DUMP_PROMPT_SETS[promptSet].questions[index];
  const seenRef = useRef(new Set<string>());

  useEffect(() => {
    if (shownOn) seenRef.current.add(`${promptSet}:${index}`);
  }, [shownOn, promptSet, index]);

  // Restarted on every advance, so a click buys the new question its full
  // time on screen.
  useEffect(() => {
    if (!shownOn || reduceMotion || isPaused) return;
    const interval = AUTO_ADVANCE_MS[shownOn];
    const timer = setInterval(() => {
      if (document.visibilityState !== "visible") return;
      setRotation(advanceFrom({ promptSet, index, screen: shownOn }, "auto"));
    }, interval);
    return () => clearInterval(timer);
  }, [shownOn, reduceMotion, isPaused, promptSet, index]);

  function advance(via: AdvanceVia) {
    if (!shownOn) return;
    setRotation(advanceFrom({ promptSet, index, screen: shownOn }, via));
  }

  function togglePause() {
    setIsPaused((paused) => !paused);
  }

  return {
    question,
    index,
    promptSet,
    advance,
    isPaused,
    togglePause,
    // Only a question the user asked for is read out: one announced on
    // every timer tick would talk over a screen reader, and into the take.
    announcement:
      current.clickedOn !== null && current.clickedOn === shownOn
        ? question.text
        : "",
    // Read when an event is sent, never while rendering.
    promptsSeen: () => seenRef.current.size,
  };
}

interface Rotation {
  promptSet: PromptSetName;
  index: number;
  // The screen "Another question" was last clicked on, until the timer
  // moves the question on.
  clickedOn: PromptScreen | null;
}

function firstRotation(promptSet: PromptSetName): Rotation {
  return { promptSet, index: FIRST_INDEX, clickedOn: null };
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
  via: AdvanceVia,
): Rotation {
  const { questions } = BRAIN_DUMP_PROMPT_SETS[from.promptSet];
  const next = (from.index + 1) % questions.length;
  trackBrainDump("brain_dump_prompt_advanced", {
    via,
    screen: from.screen,
    prompt_index: next,
    prompt_set: from.promptSet,
  });
  return {
    promptSet: from.promptSet,
    index: next,
    clickedOn: via === "click" ? from.screen : null,
  };
}
