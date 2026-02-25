import React from "react";
import { cn } from "@/lib/utils";
import { As, Variant, variantElementMap, variants } from "./helpers";

type CustomProps = {
  variant: Variant;
  as?: As;
  size?: Variant;
  className?: string;
};

export type TextProps = React.PropsWithChildren<
  CustomProps & React.ComponentPropsWithoutRef<"p">
>;

export function Text({
  children,
  variant,
  as: outerAs,
  size,
  className = "",
  ...rest
}: TextProps) {
  const variantClasses = variants[size || variant] || variants.body;
  const Element = outerAs || variantElementMap[variant];
  const combinedClassName = cn(variantClasses, className);

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
