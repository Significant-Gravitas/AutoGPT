import React from "react";
import { cn } from "@/lib/utils";
import { type Tone, tones } from "@/components/styles/tones";
import {
  As,
  textModifiers,
  Variant,
  variantElementMap,
  variants,
} from "./helpers";

type CustomProps = {
  variant: Variant;
  as?: As;
  size?: Variant;
  /** Semantic colour role; see `tones` in src/components/styles/tones.ts. */
  tone?: Tone;
  /** Font weight over the variant's own. */
  weight?: "normal" | "medium" | "semibold" | "bold";
  /** Monospace face at the variant's size, e.g. an ID or a code value. */
  mono?: boolean;
  /** Equal-width digits, for counts, prices and times that update. */
  tabular?: boolean;
  italic?: boolean;
  decoration?: "underline" | "line-through" | "none";
  transform?: "uppercase" | "lowercase" | "capitalize" | "none";
  /** Letter-spacing over the variant's own, from the tracking tokens. */
  tracking?:
    | "tight"
    | "normal"
    | "wide"
    | "widest"
    | "title"
    | "heading"
    | "display"
    | "caps"
    | "eyebrow";
  /** Line height over the variant's own: a ratio, the parent's (`inherit`),
   * or a fixed Tailwind step (`"4"` is 1rem, `"5"` 1.25rem, ...). */
  leading?:
    | "none"
    | "tight"
    | "snug"
    | "normal"
    | "relaxed"
    | "inherit"
    | "4"
    | "5"
    | "6"
    | "8";
  /** Below md, renders at body (14/20) or large (16/24) instead. */
  narrowSize?: "body" | "large";
  align?: "left" | "center" | "right" | "start" | "end";
  /** Animates colour changes. */
  motion?: "colors";
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
  weight,
  mono,
  tabular,
  italic,
  decoration,
  transform,
  tracking,
  leading,
  narrowSize,
  align,
  motion,
  className = "",
  ...rest
}: TextProps) {
  const variantClasses = variants[size || variant] || variants.body;
  const Element = outerAs || variantElementMap[variant];
  const combinedClassName = cn(
    variantClasses,
    textModifiers({
      tone,
      weight,
      mono,
      tabular,
      italic,
      decoration,
      transform,
      tracking,
      leading,
      narrowSize,
      align,
      motion,
    }),
    className,
  );

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
