import React from "react";

/**
 * PiActivityStroke icon from the stroke style in general category.
 */
interface PiActivityStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiActivityStroke({
  size = 24,
  color,
  className,
  ariaLabel = "activity icon",
  ...props
}: PiActivityStrokeProps): JSX.Element {
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
        d="M2 12h3.28a1 1 0 0 0 .948-.684L9 3l6 18 2.772-8.316a1 1 0 0 1 .949-.684H22"
        fill="none"
      />
    </svg>
  );
}
