import React from "react";

/**
 * PiLabFlaskConicalStroke icon from the stroke style in general category.
 */
interface PiLabFlaskConicalStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiLabFlaskConicalStroke({
  size = 24,
  color,
  className,
  ariaLabel = "lab-flask-conical icon",
  ...props
}: PiLabFlaskConicalStrokeProps): JSX.Element {
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
        d="M10 3h4m-4 0H9m1 0v5.523a1.5 1.5 0 0 1-.276.867l-2.96 4.179M14 3h1m-1 0v5.523c0 .31.096.613.276.867l3.8 5.364m0 0 1.63 2.301c1.172 1.656-.012 3.945-2.04 3.945H6.334c-2.028 0-3.212-2.29-2.04-3.945l2.47-3.486m11.311 1.185c-4.153 1.887-5.687-3.773-11.311-1.185"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M9 16h.01"
        fill="none"
      />
    </svg>
  );
}
