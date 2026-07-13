import React from "react";

/**
 * PiUfoStroke icon from the stroke style in general category.
 */
interface PiUfoStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiUfoStroke({
  size = 24,
  color,
  className,
  ariaLabel = "ufo icon",
  ...props
}: PiUfoStrokeProps): JSX.Element {
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
        d="M16.56 7.44C15.779 5.411 14.03 4 12 4S8.222 5.412 7.44 7.44m9.12 0C19.79 8.102 22 9.447 22 11c0 2.21-4.477 4-10 4S2 13.21 2 11c0-1.552 2.21-2.897 5.44-3.56m9.12 0a6.7 6.7 0 0 1 .404 3.1C15.5 10.831 13.807 11 12 11c-1.806 0-3.501-.168-4.964-.46a6.7 6.7 0 0 1 .403-3.1M4 17l-1 2m9-1v3m8-4 1 2"
        fill="none"
      />
    </svg>
  );
}
