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
          aria-label={IconComponent.displayName || "Icon"}
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
    aria-label="Save Icon"
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
    aria-label="Undo Icon"
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
    aria-label="Redo Icon"
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
    aria-label="Toy Brick Icon"
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
    aria-label="Circle Alert Icon"
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
    aria-label="Circle User Icon"
    {...props}
  >
    <circle cx="12" cy="12" r="10" />
    <circle cx="12" cy="10" r="3" />
    <path d="M7 20.662V19a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1.662" />
  </svg>
));

/**
 * Refresh icon component.
 *
 * @component IconRefresh
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The refresh icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconRefresh />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconRefresh className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconRefresh size="sm" onClick={handleOnClick} />
 */
export const IconRefresh = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Refresh Icon"
    {...props}
  >
    <polyline points="23 4 23 10 17 10" />
    <polyline points="1 20 1 14 7 14" />
    <path d="M3.51 9a9 9 0 0 1 14.136 -5.36L23 10" />
    <path d="M20.49 15a9 9 0 0 1 -14.136 5.36L1 14" />
  </svg>
));

/**
 * Coin icon component.
 *
 * @component IconCoin
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The coins icon.
 *
 */
export const IconCoin = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Coin Icon"
    {...props}
  >
    <circle cx="8" cy="8" r="6" />
    <path d="M18.09 10.37A6 6 0 1 1 10.34 18" />
    <path d="M7 6h1v4" />
    <path d="m16.71 13.88.7.71-2.82 2.82" />
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
    aria-label="Menu Icon"
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
    aria-label="Square Activity Icon"
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
    aria-label="Workflow Icon"
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
    aria-label="Play Icon"
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
    aria-label="Square Icon"
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
    aria-label="Package Icon"
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
    aria-label="Megaphone Icon"
    {...props}
  >
    <path d="m3 11 18-5v12L3 14v-3z" />
    <path d="M11.6 16.8a3 3 0 1 1-5.8-1.6" />
  </svg>
));

/**
 * Key icon component.
 *
 * @component IconKey
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The key icon.
 *
 * @example
 * // Default usage
 * <IconKey />
 *
 * @example
 * // With custom color and size
 * <IconKey className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconKey size="sm" onClick={handleOnClick} />
 */
export const IconKey = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Key Icon"
    {...props}
  >
    <path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z" />
    <circle cx="16.5" cy="7.5" r=".5" fill="currentColor" />
  </svg>
));

/**
 * Key(+) icon component.
 *
 * @component IconKeyPlus
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The key(+) icon.
 *
 * @example
 * // Default usage
 * <IconKeyPlus />
 *
 * @example
 * // With custom color and size
 * <IconKeyPlus className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconKeyPlus size="sm" onClick={handleOnClick} />
 */
export const IconKeyPlus = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Key Plus Icon"
    {...props}
  >
    <path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z" />
    <line x1="15.6" x2="15.6" y1="5.4" y2="11.4" />
    <line x1="12.6" x2="18.6" y1="8.4" y2="8.4" />
  </svg>
));

/**
 * User icon component.
 *
 * @component IconUser
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The user icon.
 *
 * @example
 * // Default usage
 * <IconUser />
 *
 * @example
 * // With custom color and size
 * <IconUser className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconUser size="sm" onClick={handleOnClick} />
 */
export const IconUser = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="User Icon"
    {...props}
  >
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
));

/**
 * User(+) icon component.
 *
 * @component IconUserPlus
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The user plus icon.
 *
 * @example
 * // Default usage
 * <IconUserPlus />
 *
 * @example
 * // With custom color and size
 * <IconUserPlus className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconUserPlus size="sm" onClick={handleOnClick} />
 */
export const IconUserPlus = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="User Plus Icon"
    {...props}
  >
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <line x1="19" x2="19" y1="8" y2="14" />
    <line x1="22" x2="16" y1="11" y2="11" />
  </svg>
));

/**
 * Edit icon component.
 *
 * @component IconEdit
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The edit icon.
 */
export const IconEdit = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Edit Icon"
    {...props}
  >
    <path d="M17 3.00006C17.2626 2.73741 17.5744 2.52907 17.9176 2.38693C18.2608 2.24479 18.6286 2.17163 19 2.17163C19.3714 2.17163 19.7392 2.24479 20.0824 2.38693C20.4256 2.52907 20.7374 2.73741 21 3.00006C21.2626 3.2627 21.471 3.57451 21.6131 3.91767C21.7553 4.26083 21.8284 4.62862 21.8284 5.00006C21.8284 5.37149 21.7553 5.73929 21.6131 6.08245C21.471 6.42561 21.2626 6.73741 21 7.00006L7.5 20.5001L2 22.0001L3.5 16.5001L17 3.00006Z" />
  </svg>
));

/**
 * Log out icon component.
 *
 * @component IconLogOut
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The log out icon.
 */
export const IconLogOut = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Log Out Icon"
    {...props}
  >
    <path d="M9 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H9" />
    <path d="M16 17L21 12L16 7" />
    <path d="M21 12H9" />
  </svg>
));

/**
 * Log in icon component.
 *
 * @component IconLogIn
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The log in icon.
 */
export const IconLogIn = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Log In Icon"
    {...props}
  >
    <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
    <polyline points="10 17 15 12 10 7" />
    <line x1="15" x2="3" y1="12" y2="12" />
  </svg>
));
/**
 * Settings icon component.
 *
 * @component IconSettings
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The settings icon.
 */
export const IconSettings = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Settings Icon"
    {...props}
  >
    <path d="M12.22 2H11.78C11.2496 2 10.7409 2.21071 10.3658 2.58579C9.99072 2.96086 9.78 3.46957 9.78 4V4.18C9.77964 4.53073 9.68706 4.87519 9.51154 5.17884C9.33602 5.48248 9.08374 5.73464 8.78 5.91L8.35 6.16C8.04596 6.33554 7.70108 6.42795 7.35 6.42795C6.99893 6.42795 6.65404 6.33554 6.35 6.16L6.2 6.08C5.74107 5.81526 5.19584 5.74344 4.684 5.88031C4.17217 6.01717 3.73555 6.35154 3.47 6.81L3.25 7.19C2.98526 7.64893 2.91345 8.19416 3.05031 8.706C3.18717 9.21783 3.52154 9.65445 3.98 9.92L4.13 10.02C4.43228 10.1945 4.68362 10.4451 4.85905 10.7468C5.03448 11.0486 5.1279 11.391 5.13 11.74V12.25C5.1314 12.6024 5.03965 12.949 4.86405 13.2545C4.68844 13.5601 4.43521 13.8138 4.13 13.99L3.98 14.08C3.52154 14.3456 3.18717 14.7822 3.05031 15.294C2.91345 15.8058 2.98526 16.3511 3.25 16.81L3.47 17.19C3.73555 17.6485 4.17217 17.9828 4.684 18.1197C5.19584 18.2566 5.74107 18.1847 6.2 17.92L6.35 17.84C6.65404 17.6645 6.99893 17.5721 7.35 17.5721C7.70108 17.5721 8.04596 17.6645 8.35 17.84L8.78 18.09C9.08374 18.2654 9.33602 18.5175 9.51154 18.8212C9.68706 19.1248 9.77964 19.4693 9.78 19.82V20C9.78 20.5304 9.99072 21.0391 10.3658 21.4142C10.7409 21.7893 11.2496 22 11.78 22H12.22C12.7504 22 13.2591 21.7893 13.6342 21.4142C14.0093 21.0391 14.22 20.5304 14.22 20V19.82C14.2204 19.4693 14.3129 19.1248 14.4885 18.8212C14.664 18.5175 14.9163 18.2654 15.22 18.09L15.65 17.84C15.954 17.6645 16.2989 17.5721 16.65 17.5721C17.0011 17.5721 17.346 17.6645 17.65 17.84L17.8 17.92C18.2589 18.1847 18.8042 18.2566 19.316 18.1197C19.8278 17.9828 20.2645 17.6485 20.53 17.19L20.75 16.8C21.0147 16.3411 21.0866 15.7958 20.9497 15.284C20.8128 14.7722 20.4785 14.3356 20.02 14.07L19.87 13.99C19.5648 13.8138 19.3116 13.5601 19.136 13.2545C18.9604 12.949 18.8686 12.6024 18.87 12.25V11.75C18.8686 11.3976 18.9604 11.051 19.136 10.7455C19.3116 10.4399 19.5648 10.1862 19.87 10.01L20.02 9.92C20.4785 9.65445 20.8128 9.21783 20.9497 8.706C21.0866 8.19416 21.0147 7.64893 20.75 7.19L20.53 6.81C20.2645 6.35154 19.8278 6.01717 19.316 5.88031C18.8042 5.74344 18.2589 5.81526 17.8 6.08L17.65 6.16C17.346 6.33554 17.0011 6.42795 16.65 6.42795C16.2989 6.42795 15.954 6.33554 15.65 6.16L15.22 5.91C14.9163 5.73464 14.664 5.48248 14.4885 5.17884C14.3129 4.87519 14.2204 4.53073 14.22 4.18V4C14.22 3.46957 14.0093 2.96086 13.6342 2.58579C13.2591 2.21071 12.7504 2 12.22 2Z" />
    <path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" />
  </svg>
));

/**
 * Dashboard layout icon component.
 *
 * @component IconLayoutDashboard
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The dashboard layout icon.
 */
export const IconLayoutDashboard = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
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

/**
 * Upload cloud icon component.
 *
 * @component IconUploadCloud
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The upload cloud icon.
 */
export const IconUploadCloud = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Upload Cloud Icon"
    {...props}
  >
    <path d="M4.00034 14.899C3.25738 14.1399 2.69691 13.2217 2.36137 12.214C2.02584 11.2062 1.92405 10.1353 2.0637 9.08232C2.20335 8.02938 2.5808 7.02202 3.16743 6.13655C3.75407 5.25109 4.53452 4.51074 5.44967 3.97157C6.36482 3.43241 7.39067 3.10857 8.44951 3.0246C9.50835 2.94062 10.5724 3.09871 11.5611 3.48688C12.5498 3.87505 13.4372 4.48313 14.1561 5.26506C14.8749 6.04698 15.4065 6.98225 15.7103 8.00002H17.5003C18.4659 7.99991 19.4058 8.31034 20.1813 8.88546C20.9569 9.46058 21.5269 10.2699 21.8071 11.1938C22.0874 12.1178 22.063 13.1074 21.7377 14.0164C21.4123 14.9254 20.8032 15.7057 20.0003 16.242" />
    <path d="M12 12V21" />
    <path d="M16 16L12 12L8 16" />
  </svg>
));

/**
 * Chevron up icon component.
 *
 * @component IconChevronUp
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The chevron up icon.
 */
export const IconChevronUp = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Chevron Up Icon"
    {...props}
  >
    <path d="M17 14l-5-5-5 5" />
  </svg>
));
/**
 * Marketplace icon component.
 *
 * @component IconMarketplace
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The marketplace icon.
 */
export const IconMarketplace = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Marketplace Icon"
    {...props}
  >
    <title>Marketplace</title>
    <path d="m2 7 4.41-4.41A2 2 0 0 1 7.83 2h8.34a2 2 0 0 1 1.42.59L22 7" />
    <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
    <path d="M15 22v-4a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v4" />
    <path d="M2 7h20" />
    <path d="M22 7v3a2 2 0 0 1-2 2a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 16 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 12 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 8 12a2.7 2.7 0 0 1-1.59-.63.7.7 0 0 0-.82 0A2.7 2.7 0 0 1 4 12a2 2 0 0 1-2-2V7" />
  </svg>
));

/**
 * Library icon component.
 *
 * @component IconLibrary
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The library icon.
 */
export const IconLibrary = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Library Icon"
    {...props}
  >
    <title>Library</title>
    <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20" />
  </svg>
));

export const IconStar = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Star Icon"
    {...props}
  >
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>
));

export const IconStarFilled = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="currentColor"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Star Filled Icon"
    {...props}
  >
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>
));

/**
 * Generates an array of JSX elements representing star icons based on the average rating.
 *
 * @param avgRating - The average rating (0 to 5)
 * @returns An array of star icons as JSX elements
 */
export function StarRatingIcons(avgRating: number): JSX.Element[] {
  const stars: JSX.Element[] = [];
  const rating = Math.max(0, Math.min(5, avgRating));
  for (let i = 1; i <= 5; i++) {
    if (i <= rating) {
      stars.push(<IconStarFilled key={i} className="text-black" />);
    } else {
      stars.push(<IconStar key={i} className="text-black" />);
    }
  }
  return stars;
}

/**
 * GitHub icon component.
 *
 * @component IconGithub
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The GitHub icon.
 */
export const IconGithub = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="GitHub Icon"
    {...props}
  >
    <title>GitHub</title>
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
));

/**
 * LinkedIn icon component.
 *
 * @component IconLinkedin
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The LinkedIn icon.
 */
export const IconLinkedin = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="LinkedIn Icon"
    {...props}
  >
    <title>LinkedIn</title>
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect width="4" height="12" x="2" y="9" />
    <circle cx="4" cy="4" r="2" />
  </svg>
));

/**
 * Facebook icon component.
 *
 * @component IconFacebook
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The Facebook icon.
 */
export const IconFacebook = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Facebook Icon"
    {...props}
  >
    <title>Facebook</title>
    <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
  </svg>
));

/**
 * Instagram icon component.
 *
 * @component IconInstagram
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The Instagram icon.
 */
export const IconInstagram = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Instagram Icon"
    {...props}
  >
    <title>Instagram</title>
    <rect width="20" height="20" x="2" y="2" rx="5" ry="5" />
    <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z" />
    <line x1="17.5" x2="17.51" y1="6.5" y2="6.5" />
  </svg>
));

/**
 * X (Twitter) icon component.
 *
 * @component IconX
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The X (Twitter) icon.
 */
export const IconX = createIcon((props) => (
  <svg
    role="img"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="X (Twitter) Icon"
    {...props}
  >
    <title>X (Twitter)</title>
    <path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z" />
  </svg>
));

/**
 * Medium icon component.
 *
 * @component IconMedium
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The Medium icon.
 */
export const IconMedium = createIcon((props) => (
  <svg
    role="img"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="Medium Icon"
    {...props}
  >
    <title>Medium</title>
    <path d="M13.54 12a6.8 6.8 0 01-6.77 6.82A6.8 6.8 0 010 12a6.8 6.8 0 016.77-6.82A6.8 6.8 0 0113.54 12zM20.96 12c0 3.54-1.51 6.42-3.38 6.42-1.87 0-3.39-2.88-3.39-6.42s1.52-6.42 3.39-6.42 3.38 2.88 3.38 6.42M24 12c0 3.17-.53 5.75-1.19 5.75-.66 0-1.19-2.58-1.19-5.75s.53-5.75 1.19-5.75C23.47 6.25 24 8.83 24 12z" />
  </svg>
));

/**
 * YouTube icon component.
 *
 * @component IconYoutube
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The YouTube icon.
 */
export const IconYoutube = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="lucide lucide-youtube"
    aria-label="YouTube Icon"
    {...props}
  >
    <title>YouTube</title>
    <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
    <path d="m10 15 5-3-5-3z" />
  </svg>
));

/**
 * TikTok icon component.
 *
 * @component IconTiktok
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The TikTok icon.
 */
export const IconTiktok = createIcon((props) => (
  <svg
    role="img"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="TikTok Icon"
    {...props}
  >
    <title>TikTok</title>
    <path d="M12.525.02c1.31-.02 2.61-.01 3.91-.02.08 1.53.63 3.09 1.75 4.17 1.12 1.11 2.7 1.62 4.24 1.79v4.03c-1.44-.05-2.89-.35-4.2-.97-.57-.26-1.1-.59-1.62-.93-.01 2.92.01 5.84-.02 8.75-.08 1.4-.54 2.79-1.35 3.94-1.31 1.92-3.58 3.17-5.91 3.21-1.43.08-2.86-.31-4.08-1.03-2.02-1.19-3.44-3.37-3.65-5.71-.02-.5-.03-1-.01-1.49.18-1.9 1.12-3.72 2.58-4.96 1.66-1.44 3.98-2.13 6.15-1.72.02 1.48-.04 2.96-.04 4.44-.99-.32-2.15-.23-3.02.37-.63.41-1.11 1.04-1.36 1.75-.21.51-.15 1.07-.14 1.61.24 1.64 1.82 3.02 3.5 2.87 1.12-.01 2.19-.66 2.77-1.61.19-.33.4-.67.41-1.06.1-1.79.06-3.57.07-5.36.01-4.03-.01-8.05.02-12.07z" />
  </svg>
));

/**
 * Globe icon component.
 *
 * @component IconGlobe
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The globe icon.
 */
export const IconGlobe = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="lucide lucide-globe"
    aria-label="Globe Icon"
    {...props}
  >
    <title>Globe</title>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
    <path d="M2 12h20" />
  </svg>
));

/**
 * Left Arrow icon component.
 *
 * @component IconLeftArrow
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The left arrow icon.
 */
export const IconLeftArrow = createIcon((props) => (
  <svg
    width="32"
    height="32"
    viewBox="0 0 32 32"
    strokeWidth="2"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="Left Arrow Icon"
    {...props}
  >
    <title>Left Arrow</title>
    <path
      d="M20 24L12 16L20 8"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
));

/**
 * Right Arrow icon component.
 *
 * @component IconRightArrow
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The right arrow icon.
 */
export const IconRightArrow = createIcon((props) => (
  <svg
    width="32"
    height="32"
    viewBox="0 0 32 32"
    strokeWidth="2"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="Right Arrow Icon"
    {...props}
  >
    <title>Right Arrow</title>
    <path
      d="M12 8L20 16L12 24"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
));

export const IconBuilder = createIcon((props) => <IconToyBrick {...props} />);

export enum IconType {
  Marketplace,
  Library,
  Builder,
  Edit,
  LayoutDashboard,
  UploadCloud,
  Settings,
  LogOut,
}
export function getIconForSocial(
  url: string,
  props: IconProps,
): React.ReactNode {
  const lowerCaseUrl = url.toLowerCase();

  if (lowerCaseUrl.includes("facebook.com")) {
    return <IconFacebook {...props} />;
  } else if (lowerCaseUrl.includes("twitter.com")) {
    return <IconX {...props} />;
  } else if (lowerCaseUrl.includes("x.com")) {
    return <IconX {...props} />;
  } else if (lowerCaseUrl.includes("instagram.com")) {
    return <IconInstagram {...props} />;
  } else if (lowerCaseUrl.includes("linkedin.com")) {
    return <IconLinkedin {...props} />;
  } else if (lowerCaseUrl.includes("github.com")) {
    return <IconGithub {...props} />;
  } else if (lowerCaseUrl.includes("youtube.com")) {
    return <IconYoutube {...props} />;
  } else if (lowerCaseUrl.includes("tiktok.com")) {
    return <IconTiktok {...props} />;
  } else if (lowerCaseUrl.includes("medium.com")) {
    return <IconMedium {...props} />;
  } else {
    return <IconGlobe {...props} />;
  }
}

export { iconVariants };
