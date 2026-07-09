import React from "react";

/**
 * PiChevronLeftStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiChevronLeftStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiChevronLeftStroke({
  size = 24,
  color,
  className,
  ariaLabel = "chevron-left icon",
  ...props
}: PiChevronLeftStrokeProps): JSX.Element {
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
        d="M14.125 8.14a20.4 20.4 0 0 0-3.894 3.701.47.47 0 0 0 0 .596 20.4 20.4 0 0 0 3.894 3.702"
        fill="none"
      />
    </svg>
  );
}
