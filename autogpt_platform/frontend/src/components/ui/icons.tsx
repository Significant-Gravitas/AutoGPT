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
    strokeWidth="1.25"
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
 * Shopping Cart icon component.
 *
 * @component IconShoppingCart
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The shopping cart icon.
 */
export const IconShoppingCart = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="25"
    height="24"
    viewBox="0 0 25 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Shopping Cart Icon"
    {...props}
  >
    <title>Shopping Cart</title>
    <path d="M8.8696 22C9.42188 22 9.8696 21.5523 9.8696 21C9.8696 20.4477 9.42188 20 8.8696 20C8.31731 20 7.8696 20.4477 7.8696 21C7.8696 21.5523 8.31731 22 8.8696 22Z" />
    <path d="M19.8696 22C20.4219 22 20.8696 21.5523 20.8696 21C20.8696 20.4477 20.4219 20 19.8696 20C19.3173 20 18.8696 20.4477 18.8696 21C18.8696 21.5523 19.3173 22 19.8696 22Z" />
    <path d="M2.91959 2.04999H4.91959L7.57959 14.47C7.67716 14.9248 7.93026 15.3315 8.2953 15.6198C8.66034 15.9082 9.11449 16.0603 9.57959 16.05H19.3596C19.8148 16.0493 20.2561 15.8933 20.6106 15.6078C20.9652 15.3224 21.2117 14.9245 21.3096 14.48L22.9596 7.04999H5.98959" />
  </svg>
));

/**
 * Laptop icon component.
 *
 * @component IconLaptop
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The laptop icon.
 */
export const IconLaptop = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="25"
    height="24"
    viewBox="0 0 25 24"
    fill="none"
    aria-label="Laptop Icon"
    {...props}
  >
    <title>Laptop</title>
    <g id="icon/laptop">
      <path
        id="Vector"
        d="M20.8696 16V7C20.8696 6.46957 20.6589 5.96086 20.2838 5.58579C19.9087 5.21071 19.4 5 18.8696 5H6.8696C6.33917 5 5.83046 5.21071 5.45539 5.58579C5.08032 5.96086 4.8696 6.46957 4.8696 7V16M20.8696 16H4.8696M20.8696 16L22.1496 18.55C22.2267 18.703 22.2632 18.8732 22.2556 19.0444C22.248 19.2155 22.1966 19.3818 22.1062 19.5274C22.0159 19.6729 21.8897 19.7928 21.7397 19.8756C21.5897 19.9584 21.4209 20.0012 21.2496 20H4.4896C4.31829 20.0012 4.14955 19.9584 3.99955 19.8756C3.84955 19.7928 3.72333 19.6729 3.63298 19.5274C3.54264 19.3818 3.4912 19.2155 3.4836 19.0444C3.47601 18.8732 3.51251 18.703 3.5896 18.55L4.8696 16"
      />
    </g>
  </svg>
));

/**
 * Boxes icon component.
 *
 * @component IconBoxes
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The boxes icon.
 */
export const IconBoxes = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="25"
    height="24"
    viewBox="0 0 25 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Boxes Icon"
    {...props}
  >
    <title>Boxes</title>
    <path d="M3.8396 12.92C3.54436 13.0974 3.29993 13.348 3.12999 13.6476C2.96005 13.9472 2.87035 14.2856 2.8696 14.63V17.87C2.87035 18.2144 2.96005 18.5528 3.12999 18.8524C3.29993 19.152 3.54436 19.4026 3.8396 19.58L6.8396 21.38C7.15066 21.5669 7.50671 21.6656 7.8696 21.6656C8.23249 21.6656 8.58853 21.5669 8.8996 21.38L12.8696 19V13.5L7.8696 10.5L3.8396 12.92Z" />
    <path d="M7.8696 16.5L3.1296 13.65" />
    <path d="M7.8696 16.5L12.8696 13.5" />
    <path d="M7.8696 16.5V21.67" />
    <path d="M12.8696 13.5V19L16.8396 21.38C17.1507 21.5669 17.5067 21.6656 17.8696 21.6656C18.2325 21.6656 18.5885 21.5669 18.8996 21.38L21.8996 19.58C22.1948 19.4026 22.4393 19.152 22.6092 18.8524C22.7792 18.5528 22.8688 18.2144 22.8696 17.87V14.63C22.8688 14.2856 22.7792 13.9472 22.6092 13.6476C22.4393 13.348 22.1948 13.0974 21.8996 12.92L17.8696 10.5L12.8696 13.5Z" />
    <path d="M17.8696 16.5L12.8696 13.5" />
    <path d="M17.8696 16.5L22.6096 13.65" />
    <path d="M17.8696 16.5V21.67" />
    <path d="M8.8396 4.41997C8.54436 4.59735 8.29993 4.84797 8.12999 5.14756C7.96005 5.44714 7.87035 5.78554 7.8696 6.12997V10.5L12.8696 13.5L17.8696 10.5V6.12997C17.8688 5.78554 17.7792 5.44714 17.6092 5.14756C17.4393 4.84797 17.1948 4.59735 16.8996 4.41997L13.8996 2.61997C13.5885 2.43308 13.2325 2.33435 12.8696 2.33435C12.5067 2.33435 12.1507 2.43308 11.8396 2.61997L8.8396 4.41997Z" />
    <path d="M12.8696 8.00002L8.1296 5.15002" />
    <path d="M12.8696 8.00002L17.6096 5.15002" />
    <path d="M12.8696 13.5V8" />
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
      stars.push(
        <IconStarFilled key={i} className="text-black dark:text-yellow-500" />,
      );
    } else {
      stars.push(
        <IconStar key={i} className="text-black dark:text-yellow-500" />,
      );
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
 * Close (X) icon component.
 *
 * @component IconClose
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The close icon.
 *
 * @example
 * // Default usage
 * <IconClose />
 *
 * @example
 * // With custom color and size
 * <IconClose className="text-primary" size="lg" />
 */
export const IconClose = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 14 14"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Close Icon"
    {...props}
  >
    <path d="M1 1L13 13M1 13L13 1" />
  </svg>
));

/**
 * Plus icon component.
 *
 * @component IconPlus
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The plus icon.
 *
 * @example
 * // Default usage
 * <IconPlus />
 *
 * @example
 * // With custom color and size
 * <IconPlus className="text-primary" size="lg" />
 */
export const IconPlus = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 28 28"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Plus Icon"
    {...props}
  >
    <path d="M14 5.83334V22.1667" />
    <path d="M5.83331 14H22.1666" />
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

/**
 * Person Fill icon component.
 *
 * @component IconPersonFill
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The person fill icon.
 */
export const IconPersonFill = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 53 57"
    fill="currentColor"
    aria-label="Person Fill Icon"
    {...props}
  >
    <title>Person Fill</title>
    <path d="M5.34621 56.5302C3.67955 56.5302 2.36705 56.1552 1.40871 55.4052C0.471214 54.676 0.00246429 53.6656 0.00246429 52.3739C0.00246429 50.3531 0.606631 48.2385 1.81496 46.0302C3.0233 43.801 4.7733 41.7177 7.06496 39.7802C9.35663 37.8218 12.117 36.2385 15.3462 35.0302C18.5962 33.801 22.242 33.1864 26.2837 33.1864C30.3462 33.1864 33.992 33.801 37.2212 35.0302C40.4712 36.2385 43.2316 37.8218 45.5025 39.7802C47.7941 41.7177 49.5441 43.801 50.7525 46.0302C51.9816 48.2385 52.5962 50.3531 52.5962 52.3739C52.5962 53.6656 52.117 54.676 51.1587 55.4052C50.2212 56.1552 48.9191 56.5302 47.2525 56.5302H5.34621ZM26.315 27.5927C24.0858 27.5927 22.0233 26.9885 20.1275 25.7802C18.2316 24.551 16.7004 22.9052 15.5337 20.8427C14.3879 18.7593 13.815 16.426 13.815 13.8427C13.815 11.301 14.3879 9.00932 15.5337 6.96765C16.7004 4.92598 18.2316 3.3114 20.1275 2.1239C22.0233 0.936401 24.0858 0.342651 26.315 0.342651C28.5441 0.342651 30.6066 0.925985 32.5025 2.09265C34.3983 3.25932 35.9191 4.86348 37.065 6.90515C38.2316 8.92598 38.815 11.2177 38.815 13.7802C38.815 16.3843 38.2316 18.7281 37.065 20.8114C35.9191 22.8947 34.3983 24.551 32.5025 25.7802C30.6066 26.9885 28.5441 27.5927 26.315 27.5927Z" />
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

/**
 * Integrations icon component.
 *
 * @component IconIntegrations
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The integrations icon.
 */
export const IconIntegrations = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.25"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Integrations Icon"
    {...props}
  >
    <path d="M12 20C13.6569 20 15 18.6569 15 17C15 15.3431 13.6569 14 12 14C10.3431 14 9 15.3431 9 17C9 18.6569 10.3431 20 12 20Z" />
    <path d="M4.19937 15.1C3.41858 14.3631 2.81776 13.4565 2.44336 12.4503C2.06896 11.4441 1.93102 10.3653 2.0402 9.29725C2.14939 8.22922 2.50278 7.20061 3.07304 6.29099C3.64329 5.38137 4.41514 4.6151 5.32887 4.05146C6.24261 3.48782 7.27375 3.1419 8.34254 3.04047C9.41133 2.93904 10.4891 3.0848 11.4926 3.46649C12.496 3.84818 13.3983 4.45557 14.1294 5.24168C14.8606 6.02779 15.4012 6.97155 15.7094 7.99997H17.4994C18.4532 8.01283 19.3783 8.32849 20.1411 8.90138C20.9039 9.47427 21.4649 10.2747 21.7431 11.1872C22.0213 12.0997 22.0023 13.077 21.6889 13.978C21.3755 14.879 20.7838 15.6571 19.9994 16.2" />
    <path d="M15.7008 18.4L14.8008 18.1" />
    <path d="M9.20078 15.9L8.30078 15.6" />
    <path d="M10.5996 20.7001L10.8996 19.8" />
    <path d="M13.0996 14.2L13.3996 13.3" />
    <path d="M13.5992 20.7L13.1992 19.7" />
    <path d="M10.8004 14.3L10.4004 13.3" />
    <path d="M8.30078 18.6L9.30078 18.2" />
    <path d="M14.6992 15.8L15.6992 15.4" />
  </svg>
));

/**
 * Profile icon component.
 *
 * @component IconProfile
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The profile icon.
 */
export const IconProfile = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.25"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Profile Icon"
    {...props}
  >
    <path d="M19 21V19C19 17.9391 18.5786 16.9217 17.8284 16.1716C17.0783 15.4214 16.0609 15 15 15H9C7.93913 15 6.92172 15.4214 6.17157 16.1716C5.42143 16.9217 5 17.9391 5 19V21" />
    <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" />
  </svg>
));

/**
 * Sliders icon component.
 *
 * @component IconSliders
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The sliders icon.
 *
 * @example
 * // Default usage this is the standard usage
 * <IconSliders />
 *
 * @example
 * // With custom color and size these should be used sparingly and only when necessary
 * <IconSliders className="text-primary" size="lg" />
 *
 * @example
 * // With custom size and onClick handler
 * <IconSliders size="sm" onClick={handleOnClick} />
 */
export const IconSliders = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.25"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="Sliders Icon"
    {...props}
  >
    <path d="M21 4H14" />
    <path d="M10 4H3" />
    <path d="M21 12H12" />
    <path d="M8 12H3" />
    <path d="M21 20H16" />
    <path d="M12 20H3" />
    <path d="M14 2V6" />
    <path d="M8 10V14" />
    <path d="M16 18V22" />
  </svg>
));

/**
 * More (vertical dots) icon component.
 *
 * @component IconMore
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The more options icon.
 *
 * @example
 * // Default usage
 * <IconMore />
 *
 * @example
 * // With custom color and size
 * <IconMore className="text-neutral-800" size="lg" />
 */
export const IconMore = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 20 20"
    fill="currentColor"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="More Icon"
    {...props}
  >
    <path d="M10 10.8333C10.4603 10.8333 10.8334 10.4602 10.8334 9.99999C10.8334 9.53975 10.4603 9.16666 10 9.16666C9.53978 9.16666 9.16669 9.53975 9.16669 9.99999C9.16669 10.4602 9.53978 10.8333 10 10.8333Z" />
    <path d="M10 4.99999C10.4603 4.99999 10.8334 4.6269 10.8334 4.16666C10.8334 3.70642 10.4603 3.33333 10 3.33333C9.53978 3.33333 9.16669 3.70642 9.16669 4.16666C9.16669 4.6269 9.53978 4.99999 10 4.99999Z" />
    <path d="M10 16.6667C10.4603 16.6667 10.8334 16.2936 10.8334 15.8333C10.8334 15.3731 10.4603 15 10 15C9.53978 15 9.16669 15.3731 9.16669 15.8333C9.16669 16.2936 9.53978 16.6667 10 16.6667Z" />
  </svg>
));

/**
 * External link icon component.
 *
 * @component IconExternalLink
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The external link icon.
 */
export const IconExternalLink = createIcon((props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.25"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-label="External Link Icon"
    {...props}
  >
    <path d="M18 13V19C18 19.5304 17.7893 20.0391 17.4142 20.4142C17.0391 20.7893 16.5304 21 16 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V8C3 7.46957 3.21071 6.96086 3.58579 6.58579C3.96086 6.21071 4.46957 6 5 6H11" />
    <path d="M15 3H21V9" />
    <path d="M10 14L21 3" />
  </svg>
));

export const IconSun = createIcon((props) => (
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
    {...props}
  >
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2" />
    <path d="M12 20v2" />
    <path d="m4.93 4.93 1.41 1.41" />
    <path d="m17.66 17.66 1.41 1.41" />
    <path d="M2 12h2" />
    <path d="M20 12h2" />
    <path d="m6.34 17.66-1.41 1.41" />
    <path d="m19.07 4.93-1.41 1.41" />
  </svg>
));

export const IconMoon = createIcon((props) => (
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
    {...props}
  >
    <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
  </svg>
));

/**
 * AutoGPT Logo icon component.
 *
 * @component IconAutoGPTLogo
 * @param {IconProps} props - The props object containing additional attributes and event handlers for the icon.
 * @returns {JSX.Element} - The AutoGPT logo icon.
 */
export const IconAutoGPTLogo = createIcon((props) => (
  <svg
    width="89"
    height="40"
    viewBox="0 0 89 40"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-label="AutoGPT Logo"
    {...props}
  >
    <g id="AutoGPT-logo 1" clipPath="url(#clip0_3364_2463)">
      <path
        id="Vector"
        d="M69.1364 28.8681V38.6414C69.1364 39.3617 68.5471 39.951 67.8301 39.951C67.0541 39.951 66.4124 39.4599 66.4124 38.6414V24.0584C66.4124 20.9644 68.9236 18.4531 72.0177 18.4531C75.1117 18.4531 77.623 20.9644 77.623 24.0584C77.623 27.1525 75.1117 29.6637 72.0177 29.6637C70.9634 29.6637 69.9812 29.3723 69.1397 28.8681H69.1364ZM70.2856 22.3231C71.2417 22.3231 72.0177 23.0991 72.0177 24.0552C72.0177 25.0112 71.2417 25.7872 70.2856 25.7872C70.1088 25.7872 69.9353 25.761 69.7749 25.7119C70.2824 26.3994 71.0976 26.8447 72.0177 26.8447C73.5565 26.8447 74.8039 25.5973 74.8039 24.0584C74.8039 22.5196 73.5565 21.2721 72.0177 21.2721C71.0976 21.2721 70.2824 21.7174 69.7749 22.405C69.9353 22.3559 70.1088 22.3297 70.2856 22.3297V22.3231Z"
        fill="url(#paint0_linear_3364_2463)"
      />
      <path
        id="Vector_2"
        d="M62.133 28.8675V35.144C62.133 35.7137 61.9005 36.2343 61.524 36.6075C60.6989 37.4326 59.1699 37.4326 58.3448 36.6075C57.2611 35.5238 58.2891 33.6903 56.3509 31.752C54.4126 29.8137 51.1974 29.8694 49.318 31.752C48.4504 32.6196 47.9102 33.8212 47.9102 35.144C47.9102 35.8643 48.4995 36.4536 49.2198 36.4536C49.999 36.4536 50.6375 35.9625 50.6375 35.144C50.6375 34.5743 50.87 34.057 51.2465 33.6805C52.0716 32.8554 53.6006 32.8554 54.4257 33.6805C55.6076 34.8624 54.4126 36.5289 56.4196 38.536C58.3022 40.4186 61.5731 40.4186 63.4524 38.536C64.3201 37.6683 64.8603 36.4667 64.8603 35.144V24.0545C64.8603 20.9605 62.3491 18.4492 59.255 18.4492C56.161 18.4492 53.6497 20.9605 53.6497 24.0545C53.6497 27.1486 56.161 29.6598 59.255 29.6598C60.3093 29.6598 61.2948 29.3684 62.133 28.8642V28.8675ZM59.255 26.8441C58.335 26.8441 57.5197 26.3988 57.0122 25.7112C57.1727 25.7603 57.3462 25.7865 57.523 25.7865C58.479 25.7865 59.255 25.0106 59.255 24.0545C59.255 23.0985 58.479 22.3225 57.523 22.3225C57.3462 22.3225 57.1727 22.3487 57.0122 22.3978C57.5197 21.7103 58.335 21.265 59.255 21.265C60.7938 21.265 62.0413 22.5124 62.0413 24.0512C62.0413 25.5901 60.7938 26.8375 59.255 26.8375V26.8441Z"
        fill="url(#paint1_linear_3364_2463)"
      />
      <path
        id="Vector_3"
        d="M81.709 12.959C81.709 9.51134 80.3371 6.24048 77.9045 3.80453C75.4685 1.36858 72.1977 0 68.75 0C65.3024 0 62.0315 1.37186 59.5956 3.80453C57.1596 6.24048 55.791 9.51461 55.791 12.959V13.5451C55.791 14.2948 56.4 14.9038 57.1498 14.9038C57.8996 14.9038 58.5085 14.2948 58.5085 13.5451V12.959C58.5085 10.2349 59.5956 7.64836 61.5175 5.72645C63.4394 3.80453 66.0259 2.71425 68.75 2.71425C71.4741 2.71425 74.0574 3.80126 75.9826 5.72645C77.9045 7.64836 78.9948 10.2349 78.9948 12.959C78.9948 13.7088 79.6037 14.3178 80.3535 14.3178C81.1033 14.3178 81.7123 13.7088 81.7123 12.959H81.709Z"
        fill="url(#paint2_linear_3364_2463)"
      />
      <path
        id="Vector_4"
        d="M81.7092 17.061V18.7341H83.8963C84.6232 18.7341 85.2191 19.33 85.2191 20.0569C85.2191 20.7837 84.6952 21.4582 83.8963 21.4582H81.7092V35.1964C81.7092 35.7661 81.9417 36.2834 82.3182 36.6599C83.1433 37.485 84.6723 37.485 85.4974 36.6599C85.8739 36.2834 86.1064 35.7661 86.1064 35.1964V34.738C86.1064 33.9228 86.7481 33.4284 87.5241 33.4284C88.2444 33.4284 88.8337 34.0177 88.8337 34.738V35.1964C88.8337 36.5192 88.2935 37.7208 87.4258 38.5884C85.5432 40.471 82.2822 40.471 80.3996 38.5884C79.5319 37.7208 78.9917 36.5192 78.9917 35.1964V17.061C78.9917 16.272 79.6171 15.7383 80.3832 15.7383C81.1493 15.7383 81.706 16.3342 81.706 17.061H81.7092Z"
        fill="url(#paint3_linear_3364_2463)"
      />
      <path
        id="Vector_5"
        d="M75.4293 38.6377C75.4293 39.358 74.8399 39.9441 74.1196 39.9441C73.3436 39.9441 72.7019 39.453 72.7019 38.6377V34.2013C72.7019 33.4809 73.2912 32.8916 74.0116 32.8916C74.7875 32.8916 75.4293 33.3827 75.4293 34.2013V38.6377Z"
        fill="url(#paint4_linear_3364_2463)"
      />
      <path
        id="Vector_6"
        d="M11.7672 22.2907V31.6252H8.94164V26.9399H2.82557V31.6252H0V22.2907C0 14.5998 11.7672 14.4983 11.7672 22.2907ZM44.3808 31.6252C48.5618 31.6252 51.9506 28.2365 51.9506 24.0554C51.9506 19.8744 48.5618 16.4857 44.3808 16.4857C40.1997 16.4857 36.811 19.8744 36.811 24.0554C36.811 28.2365 40.1997 31.6252 44.3808 31.6252ZM44.3808 28.7309C41.8008 28.7309 39.7086 26.6387 39.7086 24.0587C39.7086 21.4787 41.8008 19.3865 44.3808 19.3865C46.9608 19.3865 49.053 21.4787 49.053 24.0587C49.053 26.6387 46.9608 28.7309 44.3808 28.7309ZM37.3218 16.4857V19.2097H33.2095V31.6252H30.4854V19.2097H26.3731V16.4857H37.3218ZM25.0111 25.8202V16.4857H22.1855V25.8202C22.1855 30.0242 16.0661 29.9489 16.0661 25.8202V16.4857H13.2406V25.8202C13.2406 33.5111 25.0078 33.6126 25.0078 25.8202H25.0111ZM8.94164 24.2159V22.294C8.94164 18.09 2.8223 18.1653 2.8223 22.294V24.2159H8.94164Z"
        fill="#000030"
      />
      <path
        id="Vector_7"
        d="M87.4713 32.257C88.2434 32.257 88.8693 31.6311 88.8693 30.859C88.8693 30.0869 88.2434 29.4609 87.4713 29.4609C86.6992 29.4609 86.0732 30.0869 86.0732 30.859C86.0732 31.6311 86.6992 32.257 87.4713 32.257Z"
        fill="#669CF6"
      />
      <path
        id="Vector_8"
        d="M49.2167 39.9475C49.9888 39.9475 50.6147 39.3215 50.6147 38.5494C50.6147 37.7773 49.9888 37.1514 49.2167 37.1514C48.4445 37.1514 47.8186 37.7773 47.8186 38.5494C47.8186 39.3215 48.4445 39.9475 49.2167 39.9475Z"
        fill="#669CF6"
      />
    </g>
    <defs>
      <linearGradient
        id="paint0_linear_3364_2463"
        x1="62.7328"
        y1="20.9589"
        x2="62.7328"
        y2="33.2932"
        gradientUnits="userSpaceOnUse"
      >
        <stop stopColor="#000030" />
        <stop offset="1" stopColor="#9900FF" />
      </linearGradient>
      <linearGradient
        id="paint1_linear_3364_2463"
        x1="47.5336"
        y1="20.947"
        x2="47.5336"
        y2="33.2951"
        gradientUnits="userSpaceOnUse"
      >
        <stop stopColor="#000030" />
        <stop offset="1" stopColor="#4285F4" />
      </linearGradient>
      <linearGradient
        id="paint2_linear_3364_2463"
        x1="69.4138"
        y1="6.17402"
        x2="48.0898"
        y2="-3.94009"
        gradientUnits="userSpaceOnUse"
      >
        <stop stopColor="#4285F4" />
        <stop offset="1" stopColor="#9900FF" />
      </linearGradient>
      <linearGradient
        id="paint3_linear_3364_2463"
        x1="74.2976"
        y1="15.7136"
        x2="74.2976"
        y2="34.5465"
        gradientUnits="userSpaceOnUse"
      >
        <stop stopColor="#000030" />
        <stop offset="1" stopColor="#4285F4" />
      </linearGradient>
      <linearGradient
        id="paint4_linear_3364_2463"
        x1="64.3579"
        y1="24.1914"
        x2="65.0886"
        y2="30.9756"
        gradientUnits="userSpaceOnUse"
      >
        <stop stopColor="#4285F4" />
        <stop offset="1" stopColor="#9900FF" />
      </linearGradient>
      <clipPath id="clip0_3364_2463">
        <rect width="88.8696" height="40" fill="white" />
      </clipPath>
    </defs>
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
  AutoGPTLogo,
}

export function getIconForSocial(
  url: string,
  props: IconProps,
): React.ReactNode {
  const lowerCaseUrl = url.toLowerCase();
  let host;
  try {
    host = new URL(url).host;
  } catch (e) {
    return <IconGlobe {...props} />;
  }

  if (host === "facebook.com" || host.endsWith(".facebook.com")) {
    return <IconFacebook {...props} />;
  } else if (host === "twitter.com" || host.endsWith(".twitter.com")) {
    return <IconX {...props} />;
  } else if (host === "x.com" || host.endsWith(".x.com")) {
    return <IconX {...props} />;
  } else if (host === "instagram.com" || host.endsWith(".instagram.com")) {
    return <IconInstagram {...props} />;
  } else if (host === "linkedin.com" || host.endsWith(".linkedin.com")) {
    return <IconLinkedin {...props} />;
  } else if (host === "github.com" || host.endsWith(".github.com")) {
    return <IconGithub {...props} />;
  } else if (host === "youtube.com" || host.endsWith(".youtube.com")) {
    return <IconYoutube {...props} />;
  } else if (host === "tiktok.com" || host.endsWith(".tiktok.com")) {
    return <IconTiktok {...props} />;
  } else if (host === "medium.com" || host.endsWith(".medium.com")) {
    return <IconMedium {...props} />;
  } else {
    return <IconGlobe {...props} />;
  }
}

export { iconVariants };
