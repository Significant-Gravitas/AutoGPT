import { cva, type VariantProps } from "class-variance-authority";

// The look of every AttentionCard. Each variant and density renders one of
// today's cards exactly, so collapsing them is a design decision, not a
// refactor.

export const cardVariants = cva("", {
  variants: {
    variant: {
      default: "rounded-xl border border-border-base bg-surface",
      outline: "rounded-2xl bg-surface ring-1 ring-inset ring-border-base/60",
      // TODO(tokens): `foreground` holds #09090B (zinc-950), but it is the
      // shadcn text colour; there is no shadow-colour role.
      elevated:
        "overflow-hidden rounded-3xl bg-surface shadow-zinc-950 smooth-shadow-ring-sm",
      muted: "rounded-2xl border border-border bg-muted/40",
      // TODO(tokens): the stops are border-strong (#C5C5C9), foreground-subtle
      // (#ADADB3) and foreground-tertiary (#83838C), border and text roles;
      // there are no gradient-stop roles.
      featured:
        "rounded-2xl bg-gradient-to-br from-zinc-300 via-zinc-400 to-zinc-500 p-px",
      info: "rounded-2xl bg-info-surface text-body leading-5 text-info-foreground",
      success:
        "rounded-2xl bg-success-surface text-body leading-5 text-success-foreground",
      // TODO(tokens): no role for amber-800 (#92400E) text.
      warning:
        "rounded-2xl bg-warning-surface text-body leading-5 text-amber-800",
      // TODO(tokens): no role for amber-900 (#78350F) text.
      "warning-strong":
        "rounded-2xl bg-warning-surface text-left text-body leading-5 text-amber-900",
      // TODO(tokens): no role for amber-900 (#78350F) text.
      "warning-outline":
        "rounded-lg border border-warning-border bg-warning-surface text-body leading-5 text-amber-900",
      // TODO(tokens): no role for the yellow-150 border or yellow-25 surface.
      caution: "rounded-xl border border-yellow-150 bg-yellow-25",
      // TODO(tokens): no role for the red-200 (#FAC5C5) border.
      error:
        "rounded-lg border border-red-200 bg-error-surface text-caption leading-4 text-error-foreground",
    },
  },
  defaultVariants: { variant: "default" },
});

export const densityVariants = cva("", {
  variants: {
    density: {
      compact: "px-3 py-1.5",
      snug: "px-3 py-2",
      cozy: "px-3 py-2",
      default: "p-3",
      comfortable: "p-4",
      relaxed: "px-4 py-3",
      roomy: "p-5 md:p-6",
      spacious: "space-y-7 p-6",
      flush: "p-0",
    },
  },
  defaultVariants: { density: "default" },
});

export type Variant = NonNullable<VariantProps<typeof cardVariants>["variant"]>;
export type Density = NonNullable<
  VariantProps<typeof densityVariants>["density"]
>;

// The gap between a row's columns follows the card's density.
export const rowGaps: Record<Density, string> = {
  compact: "gap-1.5",
  snug: "gap-2",
  cozy: "gap-3",
  default: "gap-3",
  comfortable: "gap-3",
  relaxed: "gap-2",
  roomy: "gap-3",
  spacious: "gap-2",
  flush: "gap-3",
};

export const rowVariants = cva("flex", {
  variants: {
    align: {
      center: "items-center",
      start: "items-start",
    },
    justify: {
      start: "",
      center: "justify-center text-center",
      between: "justify-between",
    },
    // A row that sits edge to edge in a flush list card pads itself.
    inset: { true: "px-5 py-4", false: "" },
    // A row that is a link: the whole row takes the hover.
    interactive: {
      true: "group transition-colors hover:bg-surface-muted",
      false: "",
    },
  },
  defaultVariants: {
    align: "center",
    justify: "start",
    inset: false,
    interactive: false,
  },
});

// The icon's colour follows the card. Warning and error notices tint the
// whole card, so their icon inherits it. The muted and caution icons have
// never been shrink-0 and do shrink under a long unbroken name, so they keep
// that rather than move.
export const iconTints: Record<Variant, string> = {
  default: "shrink-0",
  outline: "shrink-0 text-foreground-subtle",
  elevated: "shrink-0",
  muted: "text-muted-foreground",
  featured: "shrink-0",
  info: "shrink-0",
  success: "shrink-0",
  warning: "shrink-0",
  "warning-strong": "shrink-0",
  "warning-outline": "shrink-0",
  // TODO(tokens): no role for the yellow-600 icon fill.
  caution: "fill-yellow-600 text-action-foreground",
  error: "shrink-0",
};

export const iconVariants = cva("", {
  variants: {
    // Drops a 16-18px icon onto the centre of the first 20px text line, for
    // rows aligned to the top.
    nudge: { true: "mt-0.5", false: "" },
    // A trailing chevron that brightens with the row's hover.
    trailing: {
      true: "shrink-0 self-center text-foreground-disabled transition-colors group-hover:text-foreground-tertiary",
      false: "",
    },
  },
  defaultVariants: { nudge: false, trailing: false },
});

export const actionsVariants = cva("flex gap-2", {
  variants: {
    placement: {
      // Beside the content, at the end of the row.
      inline: "shrink-0 items-center",
      // Under the content's text, inside the content column.
      below: "mt-3 flex-wrap",
      // The card's closing row of decisions.
      footer: "flex-wrap pt-2",
    },
  },
  defaultVariants: { placement: "inline" },
});
