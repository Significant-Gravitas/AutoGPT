import { cn } from "@/lib/utils";

type TintVariant = "success" | "error" | "warning" | "info";
type ChipVariant = "solid" | "secondary" | "destructive" | "outline";
type BadgeVariant = TintVariant | ChipVariant;
type BadgeSize = "small" | "medium";

interface BadgeProps {
  variant: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  className?: string;
}

// TODO(tokens): no role is emerald-50 #ECFDF5, emerald-700 #047857,
// emerald-600 #059669 or amber-800 #92400E.
const tintVariants: Record<TintVariant, string> = {
  success: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  error: "bg-error-surface text-error-foreground ring-error-hover/10",
  warning: "bg-warning-surface text-amber-800 ring-warning/20",
  info: "bg-surface-muted text-foreground-secondary ring-foreground-tertiary/10",
};

const badgeSizes: Record<BadgeSize, string> = {
  small: "px-1.5 py-0.5 text-caption-sm",
  medium: "px-2 py-0.5 text-caption leading-5",
};

// Bordered chips in a single size, which ignore `size`.
const chipBase =
  "inline-flex cursor-default items-center rounded-md border px-2.5 py-0.5 text-caption font-semibold leading-4 transition-colors focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2";

// TODO(tokens): no role is neutral-900 #171717, neutral-100 #F5F5F5 or
// neutral-200 #E5E5E5. neutral-950 #0A0A0A is the `focus` role, a ring
// colour, not a text colour.
const chipVariants: Record<ChipVariant, string> = {
  solid: "border-transparent bg-neutral-900 text-primary-foreground shadow",
  secondary: "border-transparent bg-neutral-100 text-neutral-900",
  destructive:
    "border-transparent bg-destructive text-destructive-foreground shadow",
  outline: "border-neutral-200 text-neutral-950",
};

function isChip(variant: BadgeVariant): variant is ChipVariant {
  return variant in chipVariants;
}

export function Badge({
  variant,
  size = "medium",
  children,
  className,
}: BadgeProps) {
  if (isChip(variant)) {
    return (
      <span className={cn(chipBase, chipVariants[variant], className)}>
        {children}
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md font-sans font-medium ring-1 ring-inset",
        "overflow-hidden text-ellipsis",
        badgeSizes[size],
        tintVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
