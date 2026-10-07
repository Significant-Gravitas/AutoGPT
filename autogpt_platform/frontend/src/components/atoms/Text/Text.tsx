import React from "react";
import { cn } from "@/lib/utils";
import {
  As,
  Tone,
  tones,
  Variant,
  variantElementMap,
  variants,
} from "./helpers";

type CustomProps = {
  variant: Variant;
  as?: As;
  size?: Variant;
  /**
   * Semantic colour: primary (foreground), secondary (zinc-700), muted
   * (muted-foreground, zinc-600), danger, success, warning, or inherit.
   */
  tone?: Tone;
  className?: string;
  /**
   * Adds the sentry-unmask class for static text visibility in replays.
   * Disable when rendering user-provided or dynamic content.
   */
  unmask?: boolean;
};

export type TextProps = React.PropsWithChildren<
  CustomProps & React.ComponentPropsWithoutRef<"p">
>;

export function Text({
  children,
  variant,
  as: outerAs,
  size,
  tone,
  className = "",
  unmask = true,
  ...rest
}: TextProps) {
  const variantClasses = variants[size || variant] || variants.body;
  const Element = outerAs || variantElementMap[variant];
  const combinedClassName = cn(
    variantClasses,
    tone && tones[tone],
    unmask && "sentry-unmask",
    className,
  );

  return React.createElement(
    Element,
    {
      className: combinedClassName,
      ...rest,
    },
    children,
  );
}

// Export variant names for use in stories
export const textVariants = Object.keys(variants) as Variant[];
export const textTones = Object.keys(tones) as Tone[];
export type TextVariant = Variant;
