import { useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import { BRAIN_DUMP_PROMPT_SETS, promptSetNameForRole } from "../prompts";

const SWAP_MS = 7_000;
const REDUCED_MOTION_SWAP_MS = 14_000;
// Four cards carry questions; the two quotes never change.
const QUESTION_CARDS = 4;

export function useAmbientPrompts(role: string) {
  const promptSet = promptSetNameForRole(role);
  const reduceMotion = useReducedMotion() === true;
  const [rotation, setRotation] = useState({ promptSet, swaps: 0 });
  const swaps = rotation.promptSet === promptSet ? rotation.swaps : 0;

  useEffect(() => {
    const timer = setInterval(
      () => {
        if (document.visibilityState !== "visible") return;
        setRotation((current) => ({
          promptSet,
          swaps: (current.promptSet === promptSet ? current.swaps : 0) + 1,
        }));
      },
      reduceMotion ? REDUCED_MOTION_SWAP_MS : SWAP_MS,
    );
    return () => clearInterval(timer);
  }, [promptSet, reduceMotion]);

  const { questions, examples } = BRAIN_DUMP_PROMPT_SETS[promptSet];
  return {
    questions: Array.from(
      { length: QUESTION_CARDS },
      (_, card) => questions[questionOnCard(card, swaps, questions.length)],
    ),
    examples,
    reduceMotion,
  };
}

// The cards take turns: swap n lands on card n % 4 and brings in the next
// question after the four dealt at the start.
function questionOnCard(card: number, swaps: number, total: number) {
  if (swaps <= card) return card;
  const lastSwap =
    card + QUESTION_CARDS * Math.floor((swaps - 1 - card) / QUESTION_CARDS);
  return (QUESTION_CARDS + lastSwap) % total;
}
