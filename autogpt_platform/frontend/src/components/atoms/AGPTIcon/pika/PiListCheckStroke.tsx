import React from "react";

/**
 * PiListCheckStroke icon from the stroke style in general category.
 */
interface PiListCheckStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiListCheckStroke({
  size = 24,
  color,
  className,
  ariaLabel = "list-check icon",
  ...props
}: PiListCheckStrokeProps): JSX.Element {
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
        d="M4 12h6m-6 6h6M4 6h16m-6.5 8.978 2.341 2.339a15 15 0 0 1 4.558-4.936"
        fill="none"
      />
    </svg>
  );
}
