import { cn } from "@/lib/utils";

type BadgeVariant = "success" | "error" | "warning" | "info";
type BadgeSize = "small" | "medium";

interface BadgeProps {
  variant: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  className?: string;
}

// TODO(tokens): no role is emerald-50 #ECFDF5, emerald-700 #047857,
// emerald-600 #059669 or amber-800 #92400E.
const badgeVariants: Record<BadgeVariant, string> = {
  success: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  error: "bg-error-surface text-error-foreground ring-error-hover/10",
  warning: "bg-warning-surface text-amber-800 ring-warning/20",
  info: "bg-surface-muted text-foreground-secondary ring-foreground-tertiary/10",
};

const badgeSizes: Record<BadgeSize, string> = {
  small: "px-1.5 py-0.5 text-caption-sm",
  medium: "px-2 py-0.5 text-caption leading-5",
};

export function Badge({
  variant,
  size = "medium",
  children,
  className,
}: BadgeProps) {
  return (
    <span
      data-slot="badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md font-sans font-medium ring-1 ring-inset",
        "overflow-hidden text-ellipsis",
        badgeSizes[size],
        badgeVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
