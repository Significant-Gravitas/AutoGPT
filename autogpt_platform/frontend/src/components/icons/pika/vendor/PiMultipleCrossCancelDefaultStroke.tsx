import React from "react";

/**
 * PiMultipleCrossCancelDefaultStroke icon from the stroke style in maths category.
 */
interface PiMultipleCrossCancelDefaultStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMultipleCrossCancelDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "multiple-cross-cancel-default icon",
  ...props
}: PiMultipleCrossCancelDefaultStrokeProps): JSX.Element {
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
        d="m6 18 6-6m0 0 6-6m-6 6L6 6m6 6 6 6"
        fill="none"
      />
    </svg>
  );
}
