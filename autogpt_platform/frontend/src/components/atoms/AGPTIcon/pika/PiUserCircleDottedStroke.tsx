import React from "react";

/**
 * PiUserCircleDottedStroke icon from the stroke style in users category.
 */
interface PiUserCircleDottedStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiUserCircleDottedStroke({
  size = 24,
  color,
  className,
  ariaLabel = "user-circle-dotted icon",
  ...props
}: PiUserCircleDottedStrokeProps): JSX.Element {
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
        strokeWidth="2"
        d="M12 2h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M15.826 2.761h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M8.17 2.761h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M19.065 4.924h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M4.929 4.924h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M21.226 8.154h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M2.757 8.154h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M22 12h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M2 12h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M21.22 15.872h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
        d="M2.767 15.872h.01"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M15.5 16h-7c-1.867 0-3.393 1.393-3.495 3.147A9.97 9.97 0 0 0 12 22a9.97 9.97 0 0 0 6.995-2.853C18.893 17.393 17.367 16 15.5 16Z"
        fill="none"
      />
    </svg>
  );
}
