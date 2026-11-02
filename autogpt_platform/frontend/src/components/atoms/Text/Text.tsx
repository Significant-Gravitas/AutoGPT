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
  /** Semantic colour role; see `tones` in helpers.ts for each value. */
  tone?: Tone;
  className?: string;
  /** With `as="label"`: the id of the control the label names. */
  htmlFor?: string;
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
  ...rest
}: TextProps) {
  const variantClasses = variants[size || variant] || variants.body;
  const Element = outerAs || variantElementMap[variant];
  const combinedClassName = cn(variantClasses, tone && tones[tone], className);

  return React.createElement(
    Element,
    {
      "data-slot": "text",
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
