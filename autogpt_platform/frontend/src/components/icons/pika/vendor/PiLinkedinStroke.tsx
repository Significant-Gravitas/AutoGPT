import React from "react";

/**
 * PiLinkedinStroke icon from the stroke style in apps-&-social category.
 */
interface PiLinkedinStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiLinkedinStroke({
  size = 24,
  color,
  className,
  ariaLabel = "linkedin icon",
  ...props
}: PiLinkedinStrokeProps): JSX.Element {
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
        d="M17 17v-3.5a2.5 2.5 0 0 0-5 0m0 0V17m0-3.5v-3m-4 0V17M8 7v.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M3 9.4c0-2.24 0-3.36.436-4.216a4 4 0 0 1 1.748-1.748C6.04 3 7.16 3 9.4 3h5.2c2.24 0 3.36 0 4.216.436a4 4 0 0 1 1.748 1.748C21 6.04 21 7.16 21 9.4v5.2c0 2.24 0 3.36-.436 4.216a4 4 0 0 1-1.748 1.748C17.96 21 16.84 21 14.6 21H9.4c-2.24 0-3.36 0-4.216-.436a4 4 0 0 1-1.748-1.748C3 17.96 3 16.84 3 14.6z"
        fill="none"
      />
    </svg>
  );
}
