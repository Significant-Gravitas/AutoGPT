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
  system: "bg-foreground-subtle",
  // oxlint-disable-next-line design-system/no-raw-colors -- cyan legend dot in the dev-only token badge; info is blue, a hue change deferred
  user: "bg-sky-400",
  assistant: "bg-brand/60",
  tools: "bg-success",
};
