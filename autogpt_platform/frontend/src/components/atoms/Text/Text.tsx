import React from "react";
import { As, Variant, variantElementMap, variants } from "./helpers";

type CustomProps = {
  variant: Variant;
  as?: As;
  className?: string;
};

export type TextProps = React.PropsWithChildren<
  CustomProps & React.ComponentPropsWithoutRef<"p">
>;

export function Text({
  children,
  variant,
  as: outerAs,
  className = "",
  ...rest
}: TextProps) {
  const variantClasses = variants[variant] || variants.body;
  const Element = outerAs || variantElementMap[variant];
  const combinedClassName = `${variantClasses} ${className}`.trim();

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
export type TextVariant = Variant;
