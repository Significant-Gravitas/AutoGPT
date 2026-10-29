import { cva, VariantProps } from "class-variance-authority";
import { extendedButtonVariants } from "./helpers";

// The compact family keeps the look of the deleted components/ui/button
// (SECRT-3113): rounded-md, no border, 14px/20px text and a 16px svg. Pair a
// compact-* variant with a compact-* size. It sits on the Button atom's own
// base (extendedButtonVariants with no variant and no size), so a compact
// button renders the same class string, in the same order, as it did when
// these entries lived in extendedButtonVariants.
const COMPACT_BASE =
  "gap-2 rounded-md border-0 min-w-0 text-body leading-5 focus-visible:ring-focus [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0";

export const compactButtonVariants = cva(
  extendedButtonVariants({ variant: null, size: null }),
  {
    variants: {
      variant: {
        "compact-primary": `${COMPACT_BASE} bg-action text-action-foreground shadow hover:bg-action-hover`,
        "compact-destructive": `${COMPACT_BASE} bg-error text-destructive-foreground shadow-sm hover:bg-error/90`,
        "compact-outline": `${COMPACT_BASE} border border-border-base bg-surface shadow-sm hover:bg-surface-hover hover:text-foreground-strong`,
        "compact-ghost": `${COMPACT_BASE} hover:bg-surface-hover hover:text-foreground-strong`,
      },
      size: {
        compact: "h-9 px-4 py-2",
        "compact-sm": "h-8 px-3 text-caption leading-4",
        "compact-icon": "h-9 w-9",
        "compact-icon-sm": "h-8 w-8",
      },
    },
  },
);

type CompactButtonVariant = NonNullable<
  VariantProps<typeof compactButtonVariants>["variant"]
>;

const COMPACT_BUTTON_VARIANTS: ReadonlySet<string> =
  new Set<CompactButtonVariant>([
    "compact-primary",
    "compact-destructive",
    "compact-outline",
    "compact-ghost",
  ]);

export type CompactButtonProps = Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "size"
> & {
  variant: CompactButtonVariant;
  size: NonNullable<VariantProps<typeof compactButtonVariants>["size"]>;
  ref?: React.Ref<HTMLButtonElement>;
  as?: "button";
  href?: never;
};

export function isCompactButton(props: {
  variant?: unknown;
}): props is CompactButtonProps {
  return (
    typeof props.variant === "string" &&
    COMPACT_BUTTON_VARIANTS.has(props.variant)
  );
}
