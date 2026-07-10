import React from "react";

/**
 * PiUserCircleStroke icon from the stroke style in users category.
 */
interface PiUserCircleStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiUserCircleStroke({
  size = 24,
  color,
  className,
  ariaLabel = "user-circle icon",
  ...props
}: PiUserCircleStrokeProps): JSX.Element {
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
        d="M18.995 19.147C18.893 17.393 17.367 16 15.5 16h-7c-1.867 0-3.393 1.393-3.495 3.147m13.99 0A9.97 9.97 0 0 0 22 12c0-5.523-4.477-10-10-10S2 6.477 2 12a9.97 9.97 0 0 0 3.005 7.147m13.99 0A9.97 9.97 0 0 1 12 22a9.97 9.97 0 0 1-6.995-2.853M15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
        fill="none"
      />
    </svg>
  );
}
