import React from "react";

/**
 * PiThreeDotsMenuVerticalStroke icon from the stroke style in general category.
 */
interface PiThreeDotsMenuVerticalStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiThreeDotsMenuVerticalStroke({
  size = 24,
  color,
  className,
  ariaLabel = "three-dots-menu-vertical icon",
  ...props
}: PiThreeDotsMenuVerticalStrokeProps): JSX.Element {
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
        d="M12 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M12 13a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M12 20a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"
        fill="none"
      />
    </svg>
  );
}
