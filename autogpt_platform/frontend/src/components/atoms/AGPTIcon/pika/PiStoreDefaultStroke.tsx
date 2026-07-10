import React from "react";

/**
 * PiStoreDefaultStroke icon from the stroke style in building category.
 */
interface PiStoreDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiStoreDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "store-default icon",
  ...props
}: PiStoreDefaultStrokeProps): JSX.Element {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: color || "currentColor" }}
      role="img"
      aria-label={ariaLabel}
      {...props}
    >
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M20 13.749a3.26 3.26 0 0 0 1.77-2.898c0-1.18-.22-2.351-.338-3.523-.124-1.243-.135-2.629-1.117-3.531-.997-.917-2.392-.791-3.659-.791H7.344c-1.267 0-2.662-.126-3.66.791-.98.902-.992 2.288-1.116 3.531-.117 1.172-.334 2.343-.334 3.523 0 1.261.717 2.355 1.766 2.895m16 .002a3.256 3.256 0 0 1-4.742-2.898 3.256 3.256 0 1 1-6.512 0A3.256 3.256 0 0 1 4 13.747m16 .002v4.858c0 .84 0 1.26-.163 1.58a1.5 1.5 0 0 1-.656.656c-.32.164-.74.164-1.581.164H6.4c-.84 0-1.26 0-1.581-.164a1.5 1.5 0 0 1-.656-.655C4 19.866 4 19.446 4 18.606v-4.86"
        fill="none"
      />
    </svg>
  );
}
