import React from "react";
import { As, Variant, variantElementMap, variants } from "./helpers";

type CustomProps = {
  variant: Variant;
  as?: As;
  size?: Variant;
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
  className = "",
  unmask = true,
  ...rest
}: TextProps) {
  const variantClasses = variants[size || variant] || variants.body;
  const Element = outerAs || variantElementMap[variant];
  const combinedClassName = `${variantClasses} ${
    unmask ? "sentry-unmask" : ""
  } ${className}`.trim();

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
