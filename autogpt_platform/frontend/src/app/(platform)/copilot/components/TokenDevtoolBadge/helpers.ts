import {
  AUTOCOMPACT_TOKENS,
  MODEL_CONTEXT_WINDOW,
} from "../../tokenDevtool/tokenMath";

export function windowPercent(context: number): number {
  return Math.min(100, (context / MODEL_CONTEXT_WINDOW) * 100);
}

/** Shared by both bars so the "you are about to be summarized" threshold
 *  cannot drift between them. */
export function isOverAutocompact(context: number): boolean {
  return context >= AUTOCOMPACT_TOKENS;
}

export const BREAKDOWN_COLORS = {
  // TODO(tokens): no fitting role for bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
  system: "bg-zinc-400",
  // TODO(tokens): no exact role for bg-sky-400
  user: "bg-sky-400",
  // TODO(tokens): no exact role for bg-violet-400
  assistant: "bg-violet-400",
  // TODO(tokens): no exact role for bg-emerald-400
  tools: "bg-emerald-400",
};
