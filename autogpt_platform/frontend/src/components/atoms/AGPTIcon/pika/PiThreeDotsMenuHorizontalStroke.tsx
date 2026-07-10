import React from "react";

/**
 * PiThreeDotsMenuHorizontalStroke icon from the stroke style in general category.
 */
interface PiThreeDotsMenuHorizontalStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiThreeDotsMenuHorizontalStroke({
  size = 24,
  color,
  className,
  ariaLabel = "three-dots-menu-horizontal icon",
  ...props
}: PiThreeDotsMenuHorizontalStrokeProps): JSX.Element {
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
        d="M6 12a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M13 12a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M20 12a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
        fill="none"
      />
    </svg>
  );
}
