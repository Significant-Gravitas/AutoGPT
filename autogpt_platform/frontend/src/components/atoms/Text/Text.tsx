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
  /** Semantic colour: primary zinc-900, secondary zinc-600, muted zinc-500, danger red-600. */
  tone?: Tone;
  className?: string;
  /**
   * Shows the text in session replays, which mask all text by default.
   * Only for static copy that is the same for every user.
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
  unmask = false,
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
