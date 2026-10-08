import { Progress as KobraProgress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export interface ProgressProps extends Omit<
  React.ComponentProps<typeof KobraProgress>,
  "value"
> {
  value?: number;
  max?: number;
}

// Kobra renders `role="progressbar"`, which needs an accessible name; callers
// without a visible label get a generic one unless they pass their own.
// The house bar is sized on the root (h-2 default, `h-1` etc. from callers);
// Kobra's track fills it. Colour the fill with `[&_[data-slot=progress-indicator]]:bg-*`.
export function Progress({
  className,
  value = 0,
  max = 100,
  "aria-label": ariaLabel,
  ...props
}: ProgressProps) {
  return (
    <KobraProgress
      value={value}
      max={max}
      aria-label={
        ariaLabel ?? (props["aria-labelledby"] ? undefined : "Progress")
      }
      className={cn(
        "h-2 w-full [&_[data-slot=progress-track]]:h-full",
        className,
      )}
      {...props}
    />
  );
}
