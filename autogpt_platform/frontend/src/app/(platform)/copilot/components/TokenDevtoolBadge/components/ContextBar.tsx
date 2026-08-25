import {
  AUTOCOMPACT_TOKENS,
  MODEL_CONTEXT_WINDOW,
  formatTokenCount,
} from "../../../tokenDevtool/tokenMath";
import { pct } from "../helpers";

interface Props {
  context: number;
}

export function ContextBar({ context }: Props) {
  const threshold = (AUTOCOMPACT_TOKENS / MODEL_CONTEXT_WINDOW) * 100;
  return (
    <div className="relative h-2 w-full rounded-full bg-zinc-100">
      <div
        className={
          "absolute inset-y-0 left-0 rounded-full transition-all " +
          (context >= AUTOCOMPACT_TOKENS ? "bg-amber-400" : "bg-zinc-800")
        }
        style={{ width: `${pct(context)}%` }}
      />
      <div
        className="absolute inset-y-0 w-px bg-amber-400"
        style={{ left: `${threshold}%` }}
        title={`Backend triggers summarization around ${formatTokenCount(AUTOCOMPACT_TOKENS)} tokens`}
      />
    </div>
  );
}
