"use client";

import { Slot } from "@radix-ui/react-slot";
import type { VariantProps } from "class-variance-authority";
import {
  cloneElement,
  createContext,
  isValidElement,
  useContext,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";
import { Actions } from "./components/Actions";
import { Count } from "./components/Count";
import { Dismiss } from "./components/Dismiss";
import { FeaturedSurface } from "./components/FeaturedSurface";
import { Separator } from "./components/Separator";
import {
  cardVariants,
  densityVariants,
  iconTints,
  iconVariants,
  rowGaps,
  rowVariants,
  type Density,
  type Variant,
} from "./helpers";

// The composition reference for cards and notices: an icon column, a content
// column, actions and an optional dismiss. Each slot owns its classes; the
// look of a card is `variant` (surface, border, radius, tint) and `density`
// (padding and the gap between columns) on Root, never class strings at the
// call site. The variant and density tables live in helpers.ts.

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
          <FeaturedSurface density={spacing}>
            {children.props.children}
          </FeaturedSurface>,
        )
      ) : (
        <FeaturedSurface density={spacing}>{children}</FeaturedSurface>
      );
  }

  return (
    <CardContext.Provider value={{ variant: card, density: spacing }}>
      <Comp
        data-slot="attention-card"
        className={cn(
          cardVariants({ variant: card }),
          !isFeatured && densityVariants({ density: spacing }),
          className,
        )}
        {...rest}
      >
        {content}
      </Comp>
    </CardContext.Provider>
  );
}

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
