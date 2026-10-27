import { cva, VariantProps } from "class-variance-authority";
import { extendedButtonVariants } from "./helpers";

// The compact family keeps the look of the deleted components/ui/button
// (SECRT-3113): rounded-md, no border, 14px/20px text and a 16px svg. Pair a
// compact-* variant with a compact-* size. It sits on the Button atom's own
// base (extendedButtonVariants with no variant and no size), so a compact
// button renders the same class string, in the same order, as it did when
// these entries lived in extendedButtonVariants.
// The deleted file's dark: classes are carried verbatim.
// TODO(tokens): no role holds neutral-900 #171717, neutral-100 #F5F5F5 or
// neutral-200 #E5E5E5 (border is #E4E4E7), so they stay raw. Four colours
// whose role matches in light mode stay raw too, because the role's .dark
// value differs from this button's dark colour (SECRT-3154):
// ring-neutral-950 (focus: no .dark value, dark ring is neutral-300),
// text-neutral-50 on primary (primary-foreground: .dark #18181B, dark text is
// neutral-900 #171717), bg-red-500 (error: no .dark value, dark bg is red-900)
// and bg-white (surface: no .dark value, dark bg is neutral-950).
const COMPACT_BASE =
  "gap-2 rounded-md border-0 min-w-0 text-body leading-5 focus-visible:ring-neutral-950 dark:focus-visible:ring-neutral-300 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0";

export const compactButtonVariants = cva(
  extendedButtonVariants({ variant: null, size: null }),
  {
    variants: {
      variant: {
        "compact-primary": `${COMPACT_BASE} bg-neutral-900 text-neutral-50 shadow hover:bg-neutral-900/90 dark:bg-neutral-50 dark:text-neutral-900 dark:hover:bg-neutral-50/90`,
        "compact-destructive": `${COMPACT_BASE} bg-red-500 text-destructive-foreground shadow-sm hover:bg-red-500/90 dark:bg-red-900 dark:text-neutral-50 dark:hover:bg-red-900/90`,
        "compact-outline": `${COMPACT_BASE} border border-neutral-200 bg-white shadow-sm hover:bg-neutral-100 hover:text-neutral-900 dark:border-neutral-800 dark:bg-neutral-950 dark:hover:bg-neutral-800 dark:hover:text-neutral-50`,
        "compact-ghost": `${COMPACT_BASE} hover:bg-neutral-100 hover:text-neutral-900 dark:hover:bg-neutral-800 dark:hover:text-neutral-50`,
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
