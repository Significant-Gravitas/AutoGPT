import { MODEL_CONTEXT_WINDOW } from "../../tokenDevtool/tokenMath";

export function pct(context: number): number {
  return Math.min(100, (context / MODEL_CONTEXT_WINDOW) * 100);
}

export const BREAKDOWN_COLORS = {
  system: "bg-zinc-400",
  user: "bg-sky-400",
  assistant: "bg-violet-400",
  tools: "bg-emerald-400",
};
