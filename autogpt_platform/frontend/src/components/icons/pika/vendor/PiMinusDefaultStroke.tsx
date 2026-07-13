import React from "react";

/**
 * PiMinusDefaultStroke icon from the stroke style in maths category.
 */
interface PiMinusDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMinusDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "minus-default icon",
  ...props
}: PiMinusDefaultStrokeProps): JSX.Element {
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
        d="M5 12h14"
        fill="none"
      />
    </svg>
  );
}
