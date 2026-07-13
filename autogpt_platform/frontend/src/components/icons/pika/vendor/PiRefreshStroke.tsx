import React from "react";

/**
 * PiRefreshStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiRefreshStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiRefreshStroke({
  size = 24,
  color,
  className,
  ariaLabel = "refresh icon",
  ...props
}: PiRefreshStrokeProps): JSX.Element {
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
        d="M17.5 2.474A15 15 0 0 1 18.548 6.2a.48.48 0 0 1-.297.515l-.181.07M6.5 21.527A15 15 0 0 1 5.45 17.8a.48.48 0 0 1 .297-.515l.182-.07M14.5 7.67a15 15 0 0 0 3.57-.884m0 0a8 8 0 0 0-13.912 6.797m15.75-2.79A8 8 0 0 1 5.93 17.215m3.57-.885a15 15 0 0 0-3.57.884"
        fill="none"
      />
    </svg>
  );
}
