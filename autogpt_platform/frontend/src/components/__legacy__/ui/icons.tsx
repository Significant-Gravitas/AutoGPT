import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Represents different variants of an icon, based on its size.
 */
const iconVariants = {
  size: {
    default: "size-4",
    sm: "size-2",
    lg: "size-6",
  },
} as const;

/**
 * Props for the Icon component.
 */
export interface IconProps extends React.SVGProps<SVGSVGElement> {
  size?: keyof typeof iconVariants.size;
}

/**
 * Creates an icon component that wraps a given SVG icon component.
 * This function applies consistent styling and size variants to the icon.
 *
 * @template P - The props type for the icon component
 * @param {React.FC<P>} IconComponent - The SVG icon component to be wrapped
 * @returns {React.ForwardRefExoticComponent<IconProps & React.RefAttributes<SVGSVGElement>>}
 *
 */
const createIcon = <P extends React.SVGProps<SVGSVGElement>>(
  IconComponent: React.FC<P>,
): React.ForwardRefExoticComponent<
  IconProps & React.RefAttributes<SVGSVGElement>
> => {
  const Icon = React.forwardRef<SVGSVGElement, IconProps>(
    ({ className, size = "default", ...props }, ref) => {
      return (
        <IconComponent
          className={cn(iconVariants.size[size], className)}
          ref={ref}
          {...(props as P)}
          aria-label={IconComponent.displayName || "Icon"}
        />
      );
    },
  );
  Icon.displayName = IconComponent.name || "Icon";
  return Icon;
};

/**
 * Toy brick icon component.
 *
 * @component IconToyBrick
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The toy brick icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconToyBrick />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconToyBrick className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconToyBrick size="sm" onClick={handleOnClick} />
 */
export const IconToyBrick = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Toy Brick Icon"
    {...props}
  >
    <rect width="18" height="12" x="3" y="8" rx="1" />
    <path d="M10 8V5c0-.6-.4-1-1-1H6a1 1 0 0 0-1 1v3" />
    <path d="M19 8V5c0-.6-.4-1-1-1h-3a1 1 0 0 0-1 1v3" />
  </svg>
));

/**
 * Dashboard Layout icon component.
 *
 * @component IconDashboardLayout
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The dashboard layout icon.
 */
export const IconDashboardLayout = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.25"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Dashboard Layout Icon"
    {...props}
  >
    <path d="M10 3H3V12H10V3Z" />
    <path d="M21 3H14V8H21V3Z" />
    <path d="M21 12H14V21H21V12Z" />
    <path d="M10 16H3V21H10V16Z" />
  </svg>
));

export { iconVariants };
