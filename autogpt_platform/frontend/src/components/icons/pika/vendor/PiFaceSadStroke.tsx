import React from "react";

/**
 * PiFaceSadStroke icon from the stroke style in general category.
 */
interface PiFaceSadStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFaceSadStroke({
  size = 24,
  color,
  className,
  ariaLabel = "face-sad icon",
  ...props
}: PiFaceSadStrokeProps): JSX.Element {
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
        d="M15.57 16A5 5 0 0 0 12 14.5 5 5 0 0 0 8.43 16M9 9.685v1m6-1v1M12 21.15a9.15 9.15 0 1 1 0-18.3 9.15 9.15 0 0 1 0 18.3Z"
        fill="none"
      />
    </svg>
  );
}
