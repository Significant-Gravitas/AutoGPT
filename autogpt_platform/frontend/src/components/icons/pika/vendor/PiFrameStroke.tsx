import React from "react";

/**
 * PiFrameStroke icon from the stroke style in editing category.
 */
interface PiFrameStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFrameStroke({
  size = 24,
  color,
  className,
  ariaLabel = "frame icon",
  ...props
}: PiFrameStrokeProps): JSX.Element {
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
        d="M7 3v4m0 0v10M7 7h10M7 7H3m4 10v4m0-4h10M7 17H3M21 7h-4m0 0V3m0 4v10m0 0v4m0-4h4"
        fill="none"
      />
    </svg>
  );
}
