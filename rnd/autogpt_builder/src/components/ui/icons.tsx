"use client";

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
        />
      );
    },
  );
  Icon.displayName = IconComponent.name || "Icon";
  return Icon;
};

/**
 * Save icon component.
 *
 * @component IconSave
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The save icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconSave />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconSave className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconSave size="sm" onClick={handleOnClick} />
 */
export const IconSave = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z" />
    <path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7" />
    <path d="M7 3v4a1 1 0 0 0 1 1h7" />
  </svg>
));

/**
 * Undo icon component.
 *
 * @component IconUndo2
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The undo icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconUndo2 />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconUndo2 className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconUndo2 size="sm" onClick={handleOnClick} />
 */
export const IconUndo2 = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="M9 14 4 9l5-5" />
    <path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11" />
  </svg>
));

/**
 * Redo icon component.
 *
 * @component IconRedo2
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The redo icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconRedo2 />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconRedo2 className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconRedo2 size="sm" onClick={handleOnClick} />
 */
export const IconRedo2 = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="m15 14 5-5-5-5" />
    <path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13" />
  </svg>
));

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
    {...props}
  >
    <rect width="18" height="12" x="3" y="8" rx="1" />
    <path d="M10 8V5c0-.6-.4-1-1-1H6a1 1 0 0 0-1 1v3" />
    <path d="M19 8V5c0-.6-.4-1-1-1h-3a1 1 0 0 0-1 1v3" />
  </svg>
));

/**
 * A circle alert icon component.
 *
 * @component IconCircleAlert
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The circle alert icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconCircleAlert />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconCircleAlert className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconCircleAlert size="sm" onClick={handleOnClick} />
 */
export const IconCircleAlert = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <circle cx="12" cy="12" r="10" />
    <line x1="12" x2="12" y1="8" y2="12" />
    <line x1="12" x2="12.01" y1="16" y2="16" />
  </svg>
));

/**
 * Circle User icon component.
 *
 * @component IconCircleUser
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The circle user icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconCircleUser />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconCircleUser className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconCircleUser size="sm" onClick={handleOnClick} />
 */
export const IconCircleUser = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <circle cx="12" cy="12" r="10" />
    <circle cx="12" cy="10" r="3" />
    <path d="M7 20.662V19a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1.662" />
  </svg>
));

/**
 * Menu icon component.
 *
 * @component IconMenu
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The menu icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconMenu />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconMenu className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconMenu size="sm" onClick={handleOnClick} />
 */
export const IconMenu = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <line x1="4" x2="20" y1="12" y2="12" />
    <line x1="4" x2="20" y1="6" y2="6" />
    <line x1="4" x2="20" y1="18" y2="18" />
  </svg>
));

/**
 * Square Activity icon component.
 *
 * @component IconSquareActivity
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The square activity icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconSquareActivity />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconSquareActivity className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconSquareActivity size="sm" onClick={handleOnClick} />
 */
export const IconSquareActivity = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <rect width="18" height="18" x="3" y="3" rx="2" />
    <path d="M17 12h-2l-2 5-2-10-2 5H7" />
  </svg>
));

/**
 * Workflow icon component.
 *
 * @component IconWorkFlow
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The workflow icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconWorkFlow />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconWorkFlow className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconWorkFlow size="sm" onClick={handleOnClick} />
 */
export const IconWorkFlow = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <rect width="8" height="8" x="3" y="3" rx="2" />
    <path d="M7 11v4a2 2 0 0 0 2 2h4" />
    <rect width="8" height="8" x="13" y="13" rx="2" />
  </svg>
));

/**
 * Play icon component.
 *
 * @component IconPlay
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The play icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconPlay />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconPlay className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconPlay size="sm" onClick={handleOnClick} />
 */
export const IconPlay = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <polygon points="6 3 20 12 6 21 6 3" />
  </svg>
));

/**
 * Square icon component.
 *
 * @component IconSquare
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The square icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconSquare />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconSquare className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconSquare size="sm" onClick={handleOnClick} />
 */
export const IconSquare = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <rect width="18" height="18" x="3" y="3" rx="2" />
  </svg>
));

/**
 * Package2 icon component.
 *
 * @component IconPackage2
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The package2 icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconPackage2 />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconPackage2 className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconPackage2 size="sm" onClick={handleOnClick} />
 */
export const IconPackage2 = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="M3 9h18v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9Z" />
    <path d="m3 9 2.45-4.9A2 2 0 0 1 7.24 3h9.52a2 2 0 0 1 1.8 1.1L21 9" />
    <path d="M12 3v6" />
  </svg>
));

/**
 * Megaphone icon component.
 *
 * @component IconMegaphone
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The megaphone icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconMegaphone />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconMegaphone className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconMegaphone size="sm" onClick={handleOnClick} />
 */
export const IconMegaphone = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <path d="m3 11 18-5v12L3 14v-3z" />
    <path d="M11.6 16.8a3 3 0 1 1-5.8-1.6" />
  </svg>
));

export { iconVariants };
