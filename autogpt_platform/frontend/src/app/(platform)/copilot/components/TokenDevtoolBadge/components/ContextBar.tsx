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
    <div className="relative h-2 w-full rounded-pill bg-surface-hover">
      <div
        className={
          "absolute inset-y-0 left-0 w-(--fill) rounded-pill transition-all " +
          (isOverAutocompact(context) ? "bg-warning" : "bg-action")
        }
        style={
          { "--fill": `${windowPercent(context)}%` } as React.CSSProperties
        }
      />
      <div
        aria-hidden
        className="absolute inset-y-0 left-(--threshold) w-px bg-warning"
        style={{ "--threshold": `${threshold}%` } as React.CSSProperties}
      />
    </div>
  );
}
