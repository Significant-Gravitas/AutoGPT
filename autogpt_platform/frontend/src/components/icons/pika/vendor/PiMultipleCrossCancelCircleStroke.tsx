import React from "react";

/**
 * PiMultipleCrossCancelCircleStroke icon from the stroke style in maths category.
 */
interface PiMultipleCrossCancelCircleStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMultipleCrossCancelCircleStroke({
  size = 24,
  color,
  className,
  ariaLabel = "multiple-cross-cancel-circle icon",
  ...props
}: PiMultipleCrossCancelCircleStrokeProps): JSX.Element {
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
        d="m9 15 3-3m0 0 3-3m-3 3L9 9m3 3 3 3m-3 6.15a9.15 9.15 0 1 1 0-18.3 9.15 9.15 0 0 1 0 18.3Z"
        fill="none"
      />
    </svg>
  );
}
