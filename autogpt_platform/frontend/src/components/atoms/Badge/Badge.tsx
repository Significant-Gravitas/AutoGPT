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

// TODO(tokens): no role is emerald-50 #ECFDF5, emerald-700 #047857,
// emerald-600 #059669 or amber-800 #92400E.
const tintVariants: Record<TintVariant, string> = {
  success: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  error: "bg-error-surface text-error-foreground ring-error-hover/10",
  warning: "bg-warning-surface text-amber-800 ring-warning/20",
  info: "bg-surface-muted text-foreground-secondary ring-foreground-tertiary/10",
};

const badgeSizes: Record<TintSize, string> = {
  small: "px-1.5 py-0.5 text-caption-sm",
  medium: "px-2 py-0.5 text-caption leading-5",
};

// Bordered chips, which render a <div> like the legacy badge they replace.
// Their sizes only set the type. `ring-focus` is a role that `.dark` does not
// redefine. The legacy dark: classes are moved verbatim; DS6 (SECRT-3116)
// deletes them.
const chipLayout =
  "inline-flex cursor-default items-center rounded-md border px-2.5 py-0.5";

const chipBehaviour =
  "transition-colors focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 dark:border-neutral-800 dark:focus:ring-neutral-300";

// TODO(tokens): no role is 0.75rem/1rem (text-xs), so caption plus leading-4,
// or 0.625rem at the parent's line height (text-[10px]), so tiny plus
// leading-inherit.
const chipSizes: Record<ChipSize, string> = {
  default: "text-caption font-semibold leading-4",
  tiny: "text-tiny font-semibold leading-inherit",
  "tiny-normal": "text-tiny font-normal leading-inherit",
};

// TODO(tokens): no role is neutral-900 #171717, neutral-100 #F5F5F5 or
// neutral-200 #E5E5E5. neutral-950 #0A0A0A is the `focus` role, a ring
// colour, not a text colour.
// TODO(tokens): primary-foreground holds #FAFAFA in :root but #18181B
// (24 24 27) in .dark (text on the light .dark primary fill), so it doesn't
// fit the solid chip's neutral-50.
// TODO(tokens): destructive holds #EF4444 in :root but #7F1D1D (127 29 29) in
// .dark (the dark destructive fill), so it doesn't fit the destructive chip's
// red-500. The `error` role also holds #EF4444 in both themes, but it is the
// error-state colour for text and borders, not a chip fill (rule 13 pair for
// SECRT-3154). destructive-foreground stays: it is #FAFAFA (250 250 250) in :root
// and .dark alike, the text on the destructive fill.
const chipVariants: Record<ChipVariant, string> = {
  solid:
    "border-transparent bg-neutral-900 text-neutral-50 shadow dark:bg-neutral-50 dark:text-neutral-900",
  secondary:
    "border-transparent bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-50",
  destructive:
    "border-transparent bg-red-500 text-destructive-foreground shadow dark:bg-red-900 dark:text-neutral-50",
  outline: "border-neutral-200 text-neutral-950 dark:text-neutral-50",
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
