import React from "react";

/**
 * PiFilterHorizontalStroke icon from the stroke style in general category.
 */
interface PiFilterHorizontalStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFilterHorizontalStroke({
  size = 24,
  color,
  className,
  ariaLabel = "filter-horizontal icon",
  ...props
}: PiFilterHorizontalStrokeProps): JSX.Element {
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
        d="M3 7h7m0 0a3 3 0 0 0 3 3h1a3 3 0 1 0 0-6h-1a3 3 0 0 0-3 3Zm6 10h5M20 7h1M3 17h3m0 0a3 3 0 0 0 3 3h1a3 3 0 1 0 0-6H9a3 3 0 0 0-3 3Z"
        fill="none"
      />
    </svg>
  );
}
