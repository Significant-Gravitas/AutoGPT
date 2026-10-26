import * as React from "react";
import { cn } from "@/lib/utils";

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    return (
      <div
        ref={ref}
        data-slot="progress"
        // TODO(tokens): no role is gray-200 #E5E7EB (sidebar-border has the
        // hex but belongs to the sidebar).
        className={cn(
          "relative h-2 w-full overflow-hidden rounded-full bg-gray-200",
          className,
        )}
        {...props}
      >
        <div
          data-slot="progress-indicator"
          className="w-(--progress-width) h-full bg-action transition-all duration-300 ease-in-out"
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
