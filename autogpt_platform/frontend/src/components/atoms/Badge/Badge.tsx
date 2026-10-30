import { cn } from "@/lib/utils";

type TintVariant = "success" | "error" | "warning" | "info";
type ChipVariant = "solid" | "secondary" | "destructive" | "outline";
type TintSize = "small" | "medium";
type ChipSize = "default" | "tiny" | "tiny-normal";

type BadgeVariant = TintVariant | ChipVariant;

// The chips take the chip sizes and the tints the tint sizes.
interface BadgeProps<V extends BadgeVariant> {
  variant: V;
  size?: V extends ChipVariant ? ChipSize : TintSize;
  children: React.ReactNode;
  className?: string;
}

const tintVariants: Record<TintVariant, string> = {
  success: "bg-success-surface text-success-foreground ring-success/20",
  error: "bg-error-surface text-error-foreground ring-error-hover/10",
  warning: "bg-warning-surface text-warning-foreground ring-warning/20",
  info: "bg-surface-muted text-foreground-secondary ring-foreground-tertiary/10",
};

const badgeSizes: Record<TintSize, string> = {
  small: "px-1.5 py-0.5 text-caption-sm",
  medium: "px-2 py-0.5 text-caption leading-5",
};

// Bordered chips, which render a <div> like the legacy badge they replace.
// Their sizes only set the type.
const chipLayout =
  "inline-flex cursor-default items-center rounded-md border px-2.5 py-0.5";

const chipBehaviour =
  "transition-colors focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2";

const chipSizes: Record<ChipSize, string> = {
  default: "text-caption font-semibold leading-4",
  tiny: "text-tiny font-semibold leading-inherit",
  "tiny-normal": "text-tiny font-normal leading-inherit",
};

const chipVariants: Record<ChipVariant, string> = {
  solid: "border-transparent bg-action text-action-foreground shadow",
  secondary: "border-transparent bg-surface-hover text-foreground-strong",
  destructive: "border-transparent bg-error text-destructive-foreground shadow",
  outline: "border-border-base text-foreground-base",
};

function isChip(variant: BadgeVariant): variant is ChipVariant {
  return variant in chipVariants;
}

export function Badge<V extends BadgeVariant>({
  variant: badgeVariant,
  size,
  children,
  className,
}: BadgeProps<V>) {
  const variant: BadgeVariant = badgeVariant;
  if (isChip(variant)) {
    return (
      <div
        data-slot="badge"
        className={cn(
          chipLayout,
          chipSizes[(size as ChipSize | undefined) ?? "default"],
          chipBehaviour,
          chipVariants[variant],
          className,
        )}
      >
        {children}
      </div>
    );
  }

  return (
    <span
      data-slot="badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md font-sans font-medium ring-1 ring-inset",
        "overflow-hidden text-ellipsis",
        badgeSizes[(size as TintSize | undefined) ?? "medium"],
        tintVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
