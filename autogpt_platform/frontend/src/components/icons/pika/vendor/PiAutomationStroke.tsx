import React from "react";

/**
 * PiAutomationStroke icon from the stroke style in development category.
 */
interface PiAutomationStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiAutomationStroke({
  size = 24,
  color,
  className,
  ariaLabel = "automation icon",
  ...props
}: PiAutomationStrokeProps): JSX.Element {
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
        d="M11.004 18H9A6 6 0 0 1 9 6h5.85m-3.846 12a.6.6 0 0 0-.131-.37A12.5 12.5 0 0 0 8.66 15.5m2.343 2.5c0 .13-.044.262-.131.37A12.5 12.5 0 0 1 8.66 20.5M14.85 6a3.15 3.15 0 1 0 6.3 0 3.15 3.15 0 0 0-6.3 0ZM18 21c-.932 0-1.398 0-1.765-.152a2 2 0 0 1-1.083-1.083C15 19.398 15 18.932 15 18s0-1.398.152-1.765a2 2 0 0 1 1.083-1.083C16.602 15 17.068 15 18 15s1.398 0 1.765.152a2 2 0 0 1 1.083 1.083C21 16.602 21 17.068 21 18s0 1.398-.152 1.765a2 2 0 0 1-1.083 1.083C19.398 21 18.932 21 18 21Z"
        fill="none"
      />
    </svg>
  );
}
