import * as React from "react";
import { cn } from "@/lib/utils";

// The fill's colour over `action`: a usage meter's `info`, `orange` as it
// nears a limit, `error` past it, and the quiet grey of a storage meter.
const TONES = {
  info: "[&>div]:bg-info",
  orange: "[&>div]:bg-orange",
  error: "[&>div]:bg-error",
  muted: "[&>div]:bg-fill-subtle",
} as const;

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
  tone?: keyof typeof TONES;
  /** `subtle`: a lighter track, for a slim meter beside a label. */
  track?: "subtle";
  /** Rounds the fill's leading end too, not only the track's. */
  roundedFill?: boolean;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  (
    {
      className,
      value = 0,
      max = 100,
      tone,
      track,
      roundedFill = false,
      ...props
    },
    ref,
  ) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    return (
      <div
        ref={ref}
        data-slot="progress"
        className={cn(
          "relative h-2 w-full overflow-hidden rounded-pill bg-surface-active",
          track === "subtle" && "bg-surface-hover",
          roundedFill && "[&>div]:rounded-pill",
          tone && TONES[tone],
          className,
        )}
        {...props}
      >
        <div
          data-slot="progress-indicator"
          className="h-full w-(--progress-width) bg-action transition-all duration-300 ease-in-out"
          style={
            { "--progress-width": `${percentage}%` } as React.CSSProperties
          }
        />
      </div>
    );
  },
);

Progress.displayName = "Progress";

export { Progress };
