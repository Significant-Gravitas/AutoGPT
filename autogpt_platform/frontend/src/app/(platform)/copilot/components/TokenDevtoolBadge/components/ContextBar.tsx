import {
  AUTOCOMPACT_TOKENS,
  MODEL_CONTEXT_WINDOW,
} from "../../../tokenDevtool/tokenMath";
import { isOverAutocompact, windowPercent } from "../helpers";

interface Props {
  context: number;
}

export function ContextBar({ context }: Props) {
  const threshold = (AUTOCOMPACT_TOKENS / MODEL_CONTEXT_WINDOW) * 100;
  return (
    <div className="relative h-2 w-full rounded-full bg-surface-hover">
      <div
        className={
          "absolute inset-y-0 left-0 rounded-full transition-all " +
          // TODO(tokens): no exact role for bg-amber-400
          (isOverAutocompact(context) ? "bg-amber-400" : "bg-action")
        }
        style={{ width: `${windowPercent(context)}%` }}
      />
      <div
        aria-hidden
        // TODO(tokens): no exact role for bg-amber-400
        className="absolute inset-y-0 w-px bg-amber-400"
        style={{ left: `${threshold}%` }}
      />
    </div>
  );
}
