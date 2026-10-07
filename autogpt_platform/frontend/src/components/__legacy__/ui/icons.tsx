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
  Sliders,
  Chat,
  Billing,
  Help,
  WhatsNew,
}

export { iconVariants };
