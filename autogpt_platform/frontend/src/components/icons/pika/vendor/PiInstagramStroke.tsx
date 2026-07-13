import React from "react";

/**
 * PiInstagramStroke icon from the stroke style in apps-&-social category.
 */
interface PiInstagramStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiInstagramStroke({
  size = 24,
  color,
  className,
  ariaLabel = "instagram icon",
  ...props
}: PiInstagramStrokeProps): JSX.Element {
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
        d="M17 7h.01M3 12c0-2.514 0-3.77.354-4.78a6.3 6.3 0 0 1 3.865-3.866C8.23 3 9.486 3 12 3s3.77 0 4.78.354a6.3 6.3 0 0 1 3.866 3.865C21 8.23 21 9.486 21 12s0 3.77-.354 4.78a6.3 6.3 0 0 1-3.865 3.866C15.77 21 14.514 21 12 21s-3.77 0-4.78-.354a6.3 6.3 0 0 1-3.866-3.865C3 15.77 3 14.514 3 12Zm12.778-.56a3.819 3.819 0 1 1-7.555 1.12 3.819 3.819 0 0 1 7.554-1.12Z"
        fill="none"
      />
    </svg>
  );
}
