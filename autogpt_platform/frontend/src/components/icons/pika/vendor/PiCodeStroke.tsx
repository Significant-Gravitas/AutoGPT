import React from "react";

/**
 * PiCodeStroke icon from the stroke style in development category.
 */
interface PiCodeStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiCodeStroke({
  size = 24,
  color,
  className,
  ariaLabel = "code icon",
  ...props
}: PiCodeStrokeProps): JSX.Element {
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
        d="M17 18a28.2 28.2 0 0 0 4.848-5.49.93.93 0 0 0 0-1.02A28.2 28.2 0 0 0 17 6M7.004 18a28.2 28.2 0 0 1-4.848-5.49.93.93 0 0 1 0-1.02A28.2 28.2 0 0 1 7.004 6m7-1.999-4 16"
        fill="none"
      />
    </svg>
  );
}
