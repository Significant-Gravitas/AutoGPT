import React from "react";

/**
 * PiFacebookStroke icon from the stroke style in apps-&-social category.
 */
interface PiFacebookStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFacebookStroke({
  size = 24,
  color,
  className,
  ariaLabel = "facebook icon",
  ...props
}: PiFacebookStrokeProps): JSX.Element {
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
        d="M14.596 8c-1.26 0-1.89 0-2.371.28-.423.247-.768.64-.983 1.124-.246.55-.246 1.27-.246 2.71V13m0 0v7.999m0-7.999h-2.25m2.25 0h3.75m-3.75 7.999V21m0-.001L12 21c2.514 0 3.77 0 4.78-.354a6.3 6.3 0 0 0 3.866-3.865C21 15.77 21 14.514 21 12s0-3.77-.354-4.78a6.3 6.3 0 0 0-3.865-3.866C15.77 3 14.514 3 12 3s-3.77 0-4.78.354a6.3 6.3 0 0 0-3.866 3.865C3 8.23 3 9.486 3 12s0 3.77.354 4.78a6.3 6.3 0 0 0 3.865 3.866c.866.303 1.913.347 3.777.353Z"
        fill="none"
      />
    </svg>
  );
}
