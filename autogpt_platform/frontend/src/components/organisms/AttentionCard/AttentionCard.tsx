"use client";

import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import {
  cloneElement,
  createContext,
  isValidElement,
  useContext,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon as IconAtom } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

// The composition reference for cards and notices: an icon column, a content
// column, actions and an optional dismiss. Each slot owns its classes; the
// look of a card is `variant` (surface, border, radius, tint) and `density`
// (padding and the gap between columns) on Root, never class strings at the
// call site. Every variant and density renders one of today's cards exactly,
// so collapsing them is a design decision, not a refactor.

const cardVariants = cva("", {
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

const densityVariants = cva("", {
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

type Variant = NonNullable<VariantProps<typeof cardVariants>["variant"]>;
type Density = NonNullable<VariantProps<typeof densityVariants>["density"]>;

// The gap between a row's columns follows the card's density.
const rowGaps: Record<Density, string> = {
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

const CardContext = createContext<{ variant: Variant; density: Density }>({
  variant: "default",
  density: "default",
});

interface RootProps
  extends ComponentPropsWithoutRef<"div">,
    VariantProps<typeof cardVariants>,
    VariantProps<typeof densityVariants> {
  asChild?: boolean;
}

function Root({
  variant = "default",
  density = "default",
  asChild = false,
  className,
  children,
  ...rest
}: RootProps) {
  const Comp = asChild ? Slot : "div";
  const card = variant ?? "default";
  const spacing = density ?? "default";
  const isFeatured = card === "featured";

  let content = children;
  if (isFeatured) {
    content =
      asChild && isValidElement<{ children?: ReactNode }>(children) ? (
        cloneElement(
          children,
          undefined,
          <FeaturedSurface density={density}>
            {children.props.children}
          </FeaturedSurface>,
        )
      ) : (
        <FeaturedSurface density={density}>{children}</FeaturedSurface>
      );
  }

  return (
    <CardContext.Provider value={{ variant: card, density: spacing }}>
      <Comp
        data-slot="attention-card"
        className={cn(
          cardVariants({ variant: card }),
          !isFeatured && densityVariants({ density }),
          className,
        )}
        {...rest}
      >
        {content}
      </Comp>
    </CardContext.Provider>
  );
}

interface FeaturedSurfaceProps extends VariantProps<typeof densityVariants> {
  children: ReactNode;
}

// The featured card draws its 1px gradient border as an outer fill, so its
// surface, padding and glow sit one element in.
function FeaturedSurface({ density, children }: FeaturedSurfaceProps) {
  return (
    <div
      // TODO(tokens): no radius token for 15px, the 16px card inside its 1px
      // border.
      className={cn(
        "relative overflow-hidden rounded-[15px] bg-surface",
        densityVariants({ density }),
      )}
    >
      <div
        aria-hidden
        // TODO(tokens): no token for the purple and indigo radial glow.
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_60%_at_0%_0%,rgba(168,85,247,0.10),transparent_60%),radial-gradient(120%_60%_at_100%_100%,rgba(99,102,241,0.06),transparent_60%)]"
      />
      <div className="relative">{children}</div>
    </div>
  );
}

const rowVariants = cva("flex", {
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

interface RowProps
  extends ComponentPropsWithoutRef<"div">,
    VariantProps<typeof rowVariants> {
  asChild?: boolean;
}

function Row({
  align,
  justify,
  inset,
  interactive,
  asChild = false,
  className,
  ...rest
}: RowProps) {
  const Comp = asChild ? Slot : "div";
  const { density } = useContext(CardContext);
  return (
    <Comp
      data-slot="attention-card-row"
      className={cn(
        rowVariants({ align, justify, inset, interactive }),
        rowGaps[density],
        className,
      )}
      {...rest}
    />
  );
}

// The icon's colour follows the card. Warning and error notices tint the
// whole card, so their icon inherits it. The muted and caution icons have
// never been shrink-0 and do shrink under a long unbroken name, so they keep
// that rather than move.
const iconTints: Record<Variant, string> = {
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

const iconVariants = cva("", {
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

interface IconProps
  extends ComponentPropsWithoutRef<"span">,
    VariantProps<typeof iconVariants> {}

// Wraps nothing: it hands its classes to the one icon or avatar inside.
function Icon({ nudge, trailing, className, ...rest }: IconProps) {
  const { variant } = useContext(CardContext);
  return (
    <Slot
      data-slot="attention-card-icon"
      className={cn(
        iconVariants({ nudge, trailing }),
        !trailing && iconTints[variant],
        className,
      )}
      {...rest}
    />
  );
}

interface ContentProps extends ComponentPropsWithoutRef<"div"> {
  asChild?: boolean;
}

function Content({ asChild = false, className, ...rest }: ContentProps) {
  const Comp = asChild ? Slot : "div";
  return (
    <Comp
      data-slot="attention-card-content"
      className={cn("min-w-0 flex-1", className)}
      {...rest}
    />
  );
}

const actionsVariants = cva("flex gap-2", {
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

interface ActionsProps
  extends ComponentPropsWithoutRef<"div">,
    VariantProps<typeof actionsVariants> {}

function Actions({ placement, className, ...rest }: ActionsProps) {
  return (
    <div
      data-slot="attention-card-actions"
      className={cn(actionsVariants({ placement }), className)}
      {...rest}
    />
  );
}

interface DismissProps
  extends Omit<ComponentPropsWithoutRef<"button">, "children"> {
  "aria-label": string;
  onClick: () => void;
}

function Dismiss({ className, ...rest }: DismissProps) {
  return (
    <Button
      data-slot="attention-card-dismiss"
      type="button"
      variant="icon"
      size="icon-xs"
      className={className}
      {...rest}
    >
      <IconAtom icon={Cancel01Icon} className="h-3.5 w-3.5" />
    </Button>
  );
}

// A decorative hairline: it is drawn, not announced, like the border it
// replaces.
function Separator({ className, ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div
      data-slot="attention-card-separator"
      aria-hidden
      className={cn("border-t border-separator", className)}
      {...rest}
    />
  );
}

interface CountProps extends ComponentPropsWithoutRef<"span"> {
  children: ReactNode;
}

function Count({ className, children, ...rest }: CountProps) {
  return (
    <Text
      data-slot="attention-card-count"
      variant="small"
      as="span"
      // TODO(tokens): no role for Tailwind's default gray-600; the zinc roles
      // are a different grey.
      className={cn("leading-4 text-gray-600", className)}
      {...rest}
    >
      {children}
    </Text>
  );
}

export const AttentionCard = {
  Root,
  Row,
  Icon,
  Content,
  Actions,
  Dismiss,
  Separator,
  Count,
};
