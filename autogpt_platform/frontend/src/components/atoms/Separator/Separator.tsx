import { cn } from "@/lib/utils";

type SeparatorOrientation = "horizontal" | "vertical";

interface SeparatorProps {
  /** The orientation of the separator */
  orientation?: SeparatorOrientation;
  /** Whether the separator is purely decorative (true) or represents a semantic boundary (false) */
  decorative?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * A visual separator that divides content.
 * Uses semantic `<hr>` for horizontal separators and a styled `<div>` for vertical.
 */
export function Separator({
  orientation = "horizontal",
  decorative = true,
  className,
}: SeparatorProps) {
  const baseStyles = "shrink-0 bg-neutral-200 dark:bg-neutral-800";

  if (orientation === "horizontal") {
    return (
      <hr
        className={cn(baseStyles, "h-px w-full border-0", className)}
        aria-hidden={decorative}
        role={decorative ? "none" : "separator"}
      />
    );
  }

  return (
    <div
      className={cn(baseStyles, "h-full w-px", className)}
      aria-hidden={decorative}
      role={decorative ? "none" : "separator"}
      aria-orientation="vertical"
    />
  );
}
